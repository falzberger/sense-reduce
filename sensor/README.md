# Sensor Node

This module contains the implementation for sensor nodes in the SenseReduce framework.
To sensor must implement the `AbstractSensor` interface to return a list of measurements.

## Development

To run a sensor node within the SenseReduce framework, start by creating and activating a virtual environment for its
required dependencies.

```bash
python3 -m venv venv
. venv/bin/activate
# install the minimum requirements, might need additional packages to run the applied sensor
pip3 install -r requirements.txt
```

Install the common modules from the SenseReduce framework.

```bash
pip3 install -e ..
```

To start a sensor node, you'll need to specify the sensor type and the address of the base station via the command line.
You can also change the default values for the data reduction mode, the sampling interval and the threshold.
By default, an L2ThresholdMetric is used defined to cause a violation in case the absolute difference exceeds a numeric
value. Execute `python3 main.py --help` for a full documentation.
For example:

```bash
python3 main.py sense-hat http://192.168.8.110:5000 --mode predict --interval 10 --threshold 2 
```

## Hardware Setup

Multiple temperature sensors are supported out-of-the-box (
[SenseHat](https://www.raspberrypi.com/documentation/accessories/sense-hat.html),
[DHT22](https://tutorials-raspberrypi.com/raspberry-pi-measure-humidity-temperature-dht11-dht22/),
[DS18B20](https://tutorials-raspberrypi.com/raspberry-pi-temperature-sensor-1wire-ds18b20/)).
You can test a sensor by executing `python3 temperature_sensor_<sensor>.py`, which will start continuously
monitoring and printing measurements to the console.
