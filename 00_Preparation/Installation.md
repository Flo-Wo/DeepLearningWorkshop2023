# Installation

## Anaconda environment
If you use (ana)conda, you can directly create an environment and install
the required packages by using the ``env.yml`` file attached in this folder.
Run the command
```shell
conda env create -f env.yml
```
in your shell. You can activate the environment by running the command
```shell
conda activate dl-workshop
```
in your shell

## Installation via pip (Recommended)
If you do not use conda, you can also install the required packages via pip
and the attached ``requirements.txt`` file. Create a new virtual environment and
run the command
```shell
pip install -r requirements.txt
```
in your shell.

## Testing the installation
Execute the python file ``test_imports.py`` either directly via your IDE
or by running the command
```shell
python test_imports.py
```
*Attention:* Running this file for the first time might take a few seconds.

## Final important step: installing the workshop package
In the final step, we want to install the package we will be working
on during the workshop. Navigate back to the main folder 
(i.e. ``cd ..``) and run the command
```shell
pip install -e .
```
in your shell. This will install the package located in
``./utils``. For more details check the ``setup.py`` file.

## IDE
If you want to work with an IDE, we recommend you to install Spyder (e.g. via Anaconda), Visual Studio Code
or PyCharm.

## Questions and errors
If the instructions above are not working and/or lead to errors, feel free to open
an issue on the Github page.