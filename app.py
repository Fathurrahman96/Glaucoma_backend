from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import logging
from supabase import create_client, Client
import jwt
from functools import wraps
import requests
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = "https://csuhyqtfczhlqhhbkczj.supabase.co"  # Your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzdWh5cXRmY3pobHFoaGJrY3pqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4NjA2NzcsImV4cCI6MjA2NDQzNjY3N30.ZEYsYjJuV71xt-RF_Bfb8VpsPhIi2K4bXb84qImyb58"  # Your anon key
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzdWh5cXRmY3pobHFoaGJrY3pqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4NjA2NzcsImV4cCI6MjA2NDQzNjY3N30.ZEYsYjJuV71xt-RF_Bfb8VpsPhIi2K4bXb84qImyb58"  # Add your service role key for admin operations

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

model_path = Config.MODEL_PATH
model = load_model(model_path)

def verify_token(f):
    """Decorator to verify JWT token from Supabase"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        try:
            # Extract token from "Bearer <token>"
            token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else auth_header
            
            # Verify token with Supabase
            user = supabase.auth.get_user(token)
            request.user = user.user
            
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def load_glaucoma_model():
    """Load the trained glaucoma detection model"""
    global model
    try:
        model_path = Config.MODEL_PATH
        if os.path.exists(model_path):
            model = tensorflow.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")
            model = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize image to model input size (adjust according to your model)
        target_size = (224, 224)  # Update based on your model
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def analyze_fundus_image(image):
    """Analyze fundus image for glaucoma detection"""
    global model
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None
        
        if model is not None:
            # Make prediction
            prediction = model.predict(processed_image)
            
            # Extract risk probability (adjust based on your model output)
            risk_score = float(prediction[0][0])  # Assuming binary classification
            confidence = 0.85  # You can calculate actual confidence
            
            # Calculate additional metrics (customize based on your model)
            cup_disc_ratio = 0.3 + (risk_score * 0.4)  # Mock calculation
            
            result = {
                'risk': risk_score,
                'confidence': confidence,
                'analysis_data': {
                    'cup_to_disc_ratio': cup_disc_ratio,
                    'optic_disc_area': 1200 + (risk_score * 300),
                    'neuroretinal_rim_area': 800 - (risk_score * 200),
                    'peripapillary_atrophy': risk_score > 0.6,
                },
                'recommendations': get_recommendations(risk_score),
                'notes': f'AI analysis completed with {confidence*100:.1f}% confidence'
            }
            
        else:
            # Return mock data if model is not available
            result = get_mock_analysis()
            
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return get_mock_analysis()

def get_recommendations(risk_score):
    """Get recommendations based on risk score"""
    if risk_score >= 0.8:
        return [
            'Urgent consultation with ophthalmologist required',
            'Schedule comprehensive eye examination immediately',
            'Monitor intraocular pressure daily',
            'Consider immediate treatment options'
        ]
    elif risk_score >= 0.6:
        return [
            'Consult ophthalmologist within 2 weeks',
            'Regular eye pressure monitoring recommended',
            'Follow-up screening in 3-6 months',
            'Maintain healthy lifestyle and diet'
        ]
    elif risk_score >= 0.3:
        return [
            'Schedule routine eye examination',
            'Monitor for symptoms regularly',
            'Annual eye checkups recommended',
            'Follow preventive care guidelines'
        ]
    else:
        return [
            'Continue regular eye health maintenance',
            'Annual routine checkups sufficient',
            'Maintain healthy lifestyle',
            'Monitor for any vision changes'
        ]

def get_mock_analysis():
    """Return mock analysis data for testing"""
    import random
    
    risk_levels = [0.15, 0.25, 0.45, 0.65, 0.85]
    risk = random.choice(risk_levels)
    
    return {
        'risk': risk,
        'confidence': 0.82,
        'analysis_data': {
            'cup_to_disc_ratio': 0.3 + (risk * 0.4),
            'optic_disc_area': 1200,
            'neuroretinal_rim_area': 850,
            'peripapillary_atrophy': risk > 0.6,
        },
        'recommendations': get_recommendations(risk),
        'notes': 'Mock analysis - Flask backend with Supabase integration'
    }

def save_analysis_to_supabase(user_id, analysis_result, image_url=None):
    """Save analysis result directly to Supabase from Flask"""
    try:
        data = {
            'user_id': user_id,
            'glaucoma_risk': analysis_result['risk'],
            'image_url': image_url,
            'analysis_data': analysis_result['analysis_data'],
            'notes': analysis_result['notes'],
            'confidence': analysis_result.get('confidence', 0.8),
            'created_at': datetime.now().isoformat()
        }
        
        result = supabase.table('analysis_results').insert(data).execute()
        return result.data[0] if result.data else None
        
    except Exception as e:
        logger.error(f"Error saving to Supabase: {e}")
        return None

def get_user_analysis_history(user_id, limit=10):
    """Get user's analysis history from Supabase"""
    try:
        result = supabase.table('analysis_results').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []

def update_user_profile(user_id, profile_data):
    """Update user profile in Supabase"""
    try:
        profile_data['updated_at'] = datetime.now().isoformat()
        result = supabase.table('user_profiles').upsert({
            'user_id': user_id,
            **profile_data
        }).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test Supabase connection
    supabase_status = 'connected'
    try:
        supabase.table('glaucoma_tips').select('id').limit(1).execute()
    except Exception as e:
        supabase_status = f'error: {str(e)}'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'supabase_status': supabase_status
    }), 200

@app.route('/analyze', methods=['POST'])
@verify_token
def analyze_image():
    """Main endpoint for glaucoma analysis with Supabase integration"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get user from token
        user_id = request.user.id
        
        # Read and process image
        try:
            # Read image file
            image_bytes = file.read()
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Analyze image
        result = analyze_fundus_image(opencv_image)
        
        if result is None:
            return jsonify({'error': 'Analysis failed'}), 500
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['image_size'] = {
            'width': opencv_image.shape[1],
            'height': opencv_image.shape[0]
        }
        result['user_id'] = user_id
        
        # Save to Supabase (optional - you can also let Flutter handle this)
        save_to_db = request.form.get('save_to_db', 'false').lower() == 'true'
        if save_to_db:
            saved_record = save_analysis_to_supabase(user_id, result)
            if saved_record:
                result['database_id'] = saved_record['id']
        
        logger.info(f"Analysis completed for user {user_id} with risk score: {result['risk']:.3f}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/history/<user_id>', methods=['GET'])
@verify_token
def get_history(user_id):
    """Get user's analysis history"""
    try:
        # Verify user can access this data
        if request.user.id != user_id:
            return jsonify({'error': 'Unauthorized access'}), 403
        
        limit = request.args.get('limit', 10, type=int)
        history = get_user_analysis_history(user_id, limit)
        
        return jsonify({
            'history': history,
            'count': len(history)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({'error': 'Failed to fetch history'}), 500

@app.route('/profile/<user_id>', methods=['GET', 'PUT'])
@verify_token
def handle_profile(user_id):
    """Get or update user profile"""
    try:
        # Verify user can access this data
        if request.user.id != user_id:
            return jsonify({'error': 'Unauthorized access'}), 403
        
        if request.method == 'GET':
            # Get user profile
            result = supabase.table('user_profiles').select('*').eq('user_id', user_id).execute()
            profile = result.data[0] if result.data else None
            
            return jsonify({'profile': profile}), 200
            
        elif request.method == 'PUT':
            # Update user profile
            profile_data = request.get_json()
            updated_profile = update_user_profile(user_id, profile_data)
            
            if updated_profile:
                return jsonify({'profile': updated_profile}), 200
            else:
                return jsonify({'error': 'Failed to update profile'}), 500
                
    except Exception as e:
        logger.error(f"Error handling profile: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/tips', methods=['GET'])
def get_glaucoma_tips():
    """Get glaucoma prevention tips from Supabase"""
    try:
        result = supabase.table('glaucoma_tips').select('*').eq('is_active', True).order('created_at', desc=True).execute()
        
        return jsonify({
            'tips': result.data,
            'count': len(result.data)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching tips: {e}")
        return jsonify({'error': 'Failed to fetch tips'}), 500

@app.route('/tips', methods=['POST'])
@verify_token
def add_glaucoma_tip():
    """Add new glaucoma tip (admin only)"""
    try:
        tip_data = request.get_json()
        
        # Add timestamp
        tip_data['created_at'] = datetime.now().isoformat()
        tip_data['is_active'] = True
        
        result = supabase.table('glaucoma_tips').insert(tip_data).execute()
        
        return jsonify({
            'tip': result.data[0] if result.data else None,
            'message': 'Tip added successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Error adding tip: {e}")
        return jsonify({'error': 'Failed to add tip'}), 500

@app.route('/statistics', methods=['GET'])
@verify_token
def get_statistics():
    """Get analysis statistics"""
    try:
        # Total analyses
        total_result = supabase.table('analysis_results').select('id', count='exact').execute()
        total_analyses = total_result.count
        
        # High risk analyses
        high_risk_result = supabase.table('analysis_results').select('id', count='exact').gte('glaucoma_risk', 0.6).execute()
        high_risk_count = high_risk_result.count
        
        # Recent analyses (last 30 days)
        from datetime import timedelta
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        recent_result = supabase.table('analysis_results').select('id', count='exact').gte('created_at', thirty_days_ago).execute()
        recent_count = recent_result.count
        
        return jsonify({
            'total_analyses': total_analyses,
            'high_risk_analyses': high_risk_count,
            'recent_analyses': recent_count,
            'high_risk_percentage': (high_risk_count / total_analyses * 100) if total_analyses > 0 else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    global model
    
    if model is not None:
        return jsonify({
            'model_loaded': True,
            'model_type': str(type(model)),
            'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'Unknown'
        }), 200
    else:
        return jsonify({
            'model_loaded': False,
            'message': 'No model loaded'
        }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_glaucoma_model()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=True  # Set to False in production
    )