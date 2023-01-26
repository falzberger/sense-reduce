from flask import Flask

# adapt to you needs
app = Flask('base', static_folder='models')
app.config['MODEL_DIR'] = 'models'
app.config['BASE_MODEL_UUID'] = '52918000-f986-4713-b7bb-a196db62d8de'
app.config['TRAINING_DF'] = '../simulation/zamg/zamg_vienna_hourly.pickle'
