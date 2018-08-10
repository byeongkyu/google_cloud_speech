### Install PortAudio
    $ apt-get install portaudio19-dev python-all-dev


### Install Dependencies
    $ sudo pip install -r requirements.txt


### Get Service Key of Google Cloud Speech
You can create a [Service Account](https://developers.google.com/identity/protocols/OAuth2ServiceAccount#creatinganaccount) key file. This file can be used to authenticate to Google Cloud Platform services from any environment. Get Service Key and save into config directory.


### Usage
    $ roslaunch google_cloud_speech bringup.launch


### Tips
    
You can use "echo cancel" function that was supported by pulseaudio. It is help to recognition while robot speaking. If you use bluetooth speaker, connect it. Open System Setting/Sound.

Select output device and input device, and then open Terminal:

    $ pacmd load-module module-echo-cancel aec_method=webrtc aec_args=analog_gain_control=0

To disable it,

    $ pacmd unload-module module-echo-cancel