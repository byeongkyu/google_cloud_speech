### Install PortAudio
    $ apt-get install portaudio19-dev python-all-dev

### Install Dependencies
    $ sudo pip install -r requirements.txt

### Get Service Key of Google Cloud Speech
You can create a [Service Account](https://developers.google.com/identity/protocols/OAuth2ServiceAccount#creatinganaccount) key file. This file can be used to authenticate to Google Cloud Platform services from any environment. Get Service Key and save into config directory.

### Usage
    $ roslaunch google_cloud_speech bringup.launch
