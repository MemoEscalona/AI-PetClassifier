from flask import Flask
from .config import UPLOAD_FOLDER

def create_app():
    """ Crea y configura la aplicación Flask """
    app = Flask(__name__)
    
    # Configuraciones generales
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    # Importar y registrar blueprints (si los hubiera)
    from .api import api_blueprint
    app.register_blueprint(api_blueprint, url_prefix="/api")

    return app
