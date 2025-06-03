import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://csuhyqtfczhlqhhbkczj.supabase.co')
    SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
    SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
    
    # Flask Configuration
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/glaucoma_model.h5')
    
    # Security
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key')
    
    # Image Processing
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS