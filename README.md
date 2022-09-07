# MAIZE-API: Simulation API setup

[![GPLv3 license](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/python-3.8%2B-green.svg)]()
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

The python API is available at https://github.com/axelmukwena/maize-api

#### NB: Simulations only work with data created using the BMD101 Sensor
API setup was developed in Python 3 and hosted under the directory **root_folder/maize-api**. The API is developed using the Flask Framework to receive HTTPS requests. The simulation utilizes the CNN model completes all the requirements for MA. 

After installation of all required packages and modules, obtain and copy the 'contents' of the CNN model titled **maize** saved in **maize-disease-identification/models/** and paste into **maize-api/model/**.

File structure should resemble or similar:
- maize-api
    - model
        - assets
        - variables
        - keras_metadata.pb
        - saved_model.pb
    - app
    - README.md
    - ...
  

After setup, run
```
$ python3 app.py
```
which should start up the development server at the following url  http://127.0.0.1:5000/ 


### Installing or updating packages
Just make sure env exists, otherwise create another one

- Go to folder

      $ cd folder/folder/maize-api

- Activate env

      # Mac OS
      $ source venv/bin/activate

      # Windows
      $ venv\Scripts\activate
  
- Install or update package

      # specific command
      # MacOS
      $ pip install --upgrade package

- Update *requirements.txt*

      $ pip freeze > requirements.txt

### Deactivate env
- Deactivate env

      # Mac OS
      $ deactivate

      # Windows
      $ 
