from MemSE.misc import TimingMeter
import time

def test_timingmeter():
    tm = TimingMeter('tm')
    for _ in range(10):
        with tm:
            time.sleep(1)
        print(tm.hist)
        print(tm.avg)
    assert False