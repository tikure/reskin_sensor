import atexit
import ctypes as ct
from multiprocessing import Process, Event, Pipe, Value, Array

from sensor import ReSkinBase, ReSkinSettings
from sensor_types import ReSkinData

class SensorProcess(Process):
    """
    ReSkin Sensor process. Keeps datastream running in the background.
    
    Attributes
    ----------
    sensor : ReSkinSettings
        SensorSettings object
    _pipe_in, _pipe_out : Pipe
        Two ends of a pipe for communicating with the sensor
    _sample_cnt : int
        ID samples and keep count to ensure unique samples are passed to querying 
        process
    """
    def __init__(self,sensor_settings:ReSkinSettings, 
        pipe_buffer_on_pause:bool=True, chunk_size:int=10000):
        """
        Parameters
        ----------
        sensor_settings : ReSkinSettings
            Named tuple containing settings for ReSkin sensor
        
        pipe_buffer_on_pause : bool
            Buffer will be piped internally when true.
        
        chunk_size : int
            Quantum of data piped from buffer at one time.
        """
        super(SensorProcess, self).__init__()
        self.device_id = sensor_settings.device_id

        self._pipe_in, self._pipe_out = Pipe()
        self._sample_cnt = Value(ct.c_uint64)
        self._buffer_size = Value(ct.c_uint64)
        
        self._last_time = Value(ct.c_double)
        self._last_delay = Value(ct.c_double)
        self._last_reading = Array(ct.c_float, sensor_settings.num_mags * 4)

        self.sensor_settings = sensor_settings

        self.pipe_buffer_on_pause = pipe_buffer_on_pause
        self._chunk_size = chunk_size

        self._event_is_streaming = Event()
        self._event_quit_request = Event()
        self._event_sending_data = Event()

        atexit.register(self.join)
        pass

    # Might need to make sure we aren't passing mutable object.
    @property
    def last_reading(self):
        return ReSkinData(
            time = self._last_time.value,
            acquisition_delay=self._last_delay.value,
            data = self._last_reading[:],
            device_id=self.device_id)
        # return self._last_reading
    
    @property
    def sample_cnt(self):
        return self._sample_cnt.value

    def start_streaming(self):
        self._event_is_streaming.set()

    def pause_streaming(self):
        self._event_is_streaming.clear()

    def get_samples(self, num_samples=5):
        """
        Return a specified number of samples from the ReSkin Sensor
        
        Parameters
        ----------
        num_samples : int
            Number of samples required
        """
        # Only sends samples if streaming is on. Sends empty list otherwise.
        if not self._event_is_streaming.is_set():
            return []
        samples = []
        if num_samples <=0:
            return samples
        last_cnt = self._sample_cnt.value
        samples = [self.last_reading]
        while len(samples) < num_samples:
            # print(self._sample_cnt.value)
            if last_cnt == self._sample_cnt.value:
                continue
            last_cnt = self._sample_cnt.value
            samples.append(self.last_reading)
        
        return samples

    def get_buffer(self, timeout=1.0):
        """
        Return the recorded buffer
        
        Parameters
        ----------
        timeout : int
            Time to wait for data to start getting piped.
        """
        rtn = []
        if self._event_sending_data.is_set() or self._buffer_size.value > 0:
            self._event_sending_data.wait(timeout=timeout)
            while self._pipe_in.poll() or self._buffer_size.value > 0:
                rtn.extend(self._pipe_in.recv())
            self._event_sending_data.clear()
        
        return rtn

    def join(self, timeout=None):
        """
        Clean up before exiting
        """
        self._event_quit_request.set()
        # self.sensor.close()
        super(SensorProcess, self).join(timeout)
    
    def run(self):
        """
        This loop runs until it's asked to quit.
        """
        buffer = []
        # Initialize sensor
        print(self._last_reading)
        self.sensor = ReSkinBase(
            num_mags=self.sensor_settings.num_mags,
            port=self.sensor_settings.port,
            baudrate=self.sensor_settings.baudrate,
            burst_mode=self.sensor_settings.burst_mode,
            device_id=self.sensor_settings.device_id)
        
        self.sensor._initialize()
        is_streaming = False

        while not self._event_quit_request.is_set():
            if self._event_is_streaming.is_set():
                if not is_streaming:
                    is_streaming = True
                    # Any logging or stuff you want to do when polling has
                    # just started should go here
                self._last_time.value, self._last_delay.value, \
                    self._last_reading[:] = self.sensor.get_data()

                self._sample_cnt.value += 1

                buffer.append(self.last_reading)
                self._buffer_size.value = len(buffer)

            else:
                if is_streaming:
                    is_streaming = False

                if self.pipe_buffer_on_pause and self._buffer_size.value > 0:
                    self._event_sending_data.set()
                    chk = self._chunk_size
                    while len(buffer)>0:
                        if chk > len(buffer):
                            chk = len(buffer)
                        self._pipe_out.send(buffer[0:chk])
                        buffer[0:chk] = []
                        self._buffer_size.value = len(buffer)

if __name__ == '__main__':
    test_settings = ReSkinSettings(
        num_mags=5,
        port="COM32",
        baudrate=115200,
        burst_mode=True,
        device_id=1
    )
    # test_sensor = ReSkinBase(5, port="COM32", baudrate=115200)
    test_proc = SensorProcess(test_settings, pipe_buffer_on_pause=True)
    test_proc.start()
    

    test_proc.start_streaming()
    import time
    time.sleep(2.0)
    print(test_proc.get_samples(100))
    test_proc.pause_streaming()
    
    print(len(test_proc.get_buffer()))
    print(test_proc.last_reading)
    
    while True:
        input('aaaa')