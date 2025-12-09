from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

def init_ee():
    """Initialize Google Earth Engine with service account"""
    try:
        import os, json, tempfile
        
        # Get credentials from environment variable
        creds_json = os.environ.get('EE_CREDENTIALS')
        if not creds_json:
            print(⚠️ No EE_CREDENTIALS environment variable found")
            return False
            
        # Parse the JSON
        creds = json.loads(creds_json)
        
        # Save to temporary file (handles \n in private key)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(creds, f, indent=2)
            temp_file = f.name
        
        try:
            # Initialize Earth Engine with the temp file
            credentials = ee.ServiceAccountCredentials(
                creds['client_email'],  # Service account email
                temp_file               # Path to credentials file
            )
            ee.Initialize(credentials)
            print(f"✓ Earth Engine initialized with project: {creds.get('project_id', 'unknown')}")
            return True
        finally:
            # Clean up temp file
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {str(e)[:100]}")  # Truncate long errors
        return False

# ... rest of the backend code stays the same ...

# Load PyTorch Model
class DenseModel(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc_out = nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc_out(x)
        return x

def load_model():
    """Load the trained balanced model"""
    model_path = 'satellite_risk_assessment_0437/models/balanced_model_checkpoint.pt'
    device = 'cpu'
    
    model = DenseModel(input_dim=16)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    return model, device

# Global model instance
MODEL = None
DEVICE = None
EE_INITIALIZED = False

def get_weather_data(lat, lng):
    """
    Get actual temperature and precipitation data from ERA5 climate reanalysis
    via Google Earth Engine
    
    Args:
        lat: Latitude
        lng: Longitude
    
    Returns:
        dict with temperature and precipitation
    """
    try:
        point = ee.Geometry.Point([lng, lat])
        
        # Get current date and 30 days back
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # ERA5 Daily Aggregates - Temperature (in Kelvin)
        era5_temp = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filterBounds(point) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .select('temperature_2m')
        
        # ERA5 Daily Aggregates - Precipitation (in meters)
        era5_precip = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filterBounds(point) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .select('total_precipitation_sum')
        
        # Calculate mean temperature over period
        if era5_temp.size().getInfo() > 0:
            temp_mean = era5_temp.mean()
            temp_sample = temp_mean.sample(region=point, scale=11132, numPixels=1).first().getInfo()
            temp_kelvin = temp_sample['properties'].get('temperature_2m', 288.15)
            temp_celsius = temp_kelvin - 273.15
            temp_fahrenheit = (temp_celsius * 9/5) + 32
        else:
            # Fallback to MODIS Land Surface Temperature if ERA5 unavailable
            modis_temp = ee.ImageCollection('MODIS/006/MOD11A1') \
                .filterBounds(point) \
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                .select('LST_Day_1km')
            
            if modis_temp.size().getInfo() > 0:
                temp_mean = modis_temp.mean().multiply(0.02).subtract(273.15)  # Scale and convert to Celsius
                temp_sample = temp_mean.sample(region=point, scale=1000, numPixels=1).first().getInfo()
                temp_celsius = temp_sample['properties'].get('LST_Day_1km', 15.0)
                temp_fahrenheit = (temp_celsius * 9/5) + 32
            else:
                temp_fahrenheit = None
        
        # Calculate total precipitation over 30 days (convert meters to mm)
        if era5_precip.size().getInfo() > 0:
            precip_sum = era5_precip.sum()
            precip_sample = precip_sum.sample(region=point, scale=11132, numPixels=1).first().getInfo()
            precip_meters = precip_sample['properties'].get('total_precipitation_sum', 0)
            precip_mm = precip_meters * 1000  # Convert to mm
        else:
            # Fallback to CHIRPS precipitation
            chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterBounds(point) \
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if chirps.size().getInfo() > 0:
                precip_sum = chirps.sum()
                precip_sample = precip_sum.sample(region=point, scale=5566, numPixels=1).first().getInfo()
                precip_mm = precip_sample['properties'].get('precipitation', 0)
            else:
                precip_mm = None
        
        return {
            'temperature_fahrenheit': round(temp_fahrenheit, 1) if temp_fahrenheit else None,
            'temperature_celsius': round(temp_celsius, 1) if temp_fahrenheit else None,
            'precipitation_mm_30day': round(precip_mm, 1) if precip_mm is not None else None,
            'data_source': 'ERA5_LAND' if era5_precip.size().getInfo() > 0 else 'CHIRPS/MODIS',
            'period_days': 30
        }
    
    except Exception as e:
        print(f"Warning: Could not fetch weather data: {e}")
        return {
            'temperature_fahrenheit': None,
            'temperature_celsius': None,
            'precipitation_mm_30day': None,
            'data_source': 'unavailable',
            'error': str(e)
        }

def get_satellite_features(lat, lng, buffer_km=5):
    """
    Fetch satellite imagery and compute features for the given location
    
    Args:
        lat: Latitude
        lng: Longitude
        buffer_km: Buffer radius in kilometers
    
    Returns:
        dict with features and image
    """
    if not EE_INITIALIZED:
        raise Exception("Earth Engine not initialized")
    
    # Define area of interest
    point = ee.Geometry.Point([lng, lat])
    region = point.buffer(buffer_km * 1000)  # Convert km to meters
    
    # Date range (last 90 days for better cloud-free coverage)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Get Sentinel-2 Surface Reflectance imagery
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    
    collection_size = s2.size().getInfo()
    if collection_size == 0:
        # Try with higher cloud threshold
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        
        collection_size = s2.size().getInfo()
        if collection_size == 0:
            raise Exception("No Sentinel-2 imagery available for this location in the past 90 days")
    
    # Get median composite to reduce cloud影響
    image = s2.median()
    
    # Compute spectral indices
    # NDVI: Normalized Difference Vegetation Index
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # NDWI: Normalized Difference Water Index
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    # NDBI: Normalized Difference Built-up Index
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    
    # NBR: Normalized Burn Ratio (for wildfire assessment)
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    
    # NDMI: Normalized Difference Moisture Index
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    # Get SRTM elevation data (90m resolution)
    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.select('elevation')
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    
    # Combine all bands
    combined = image.addBands([ndvi, ndwi, ndbi, nbr, ndmi, elevation, slope, aspect])
    
    # Sample features in the region (multiple points for robustness)
    sample = combined.sample(
        region=region, 
        scale=30,  # 30m resolution
        numPixels=100,
        seed=42,
        geometries=True
    ).getInfo()
    
    if not sample['features'] or len(sample['features']) == 0:
        raise Exception("Could not extract features from imagery - area may be over water or outside coverage")
    
    # Calculate mean and std for each band
    features_list = sample['features']
    feature_dict = {}
    
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI', 'NBR', 'NDMI', 'elevation', 'slope', 'aspect']
    
    for band in bands:
        values = [f['properties'].get(band) for f in features_list if band in f['properties'] and f['properties'].get(band) is not None]
        if values:
            feature_dict[band] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        else:
            feature_dict[band] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
    
    # Generate true color visualization
    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'gamma': 1.4
    }
    
    # Get image thumbnail
    try:
        thumbnail_url = image.getThumbURL({
            'region': region.getInfo()['coordinates'],
            'dimensions': 512,
            'format': 'png',
            **vis_params
        })
    except Exception as e:
        print(f"Warning: Could not generate thumbnail: {e}")
        thumbnail_url = None
    
    # Get acquisition date of most recent image
    most_recent = s2.sort('system:time_start', False).first()
    acquisition_date = datetime.fromtimestamp(
        most_recent.get('system:time_start').getInfo() / 1000
    ).strftime('%Y-%m-%d')
    
    return {
        'features': feature_dict,
        'image_url': thumbnail_url,
        'sample_size': len(features_list),
        'imagery_count': collection_size,
        'acquisition_date': acquisition_date,
        'date_range': {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
    }

def prepare_model_input(features, weather):
    """
    Convert raw features to model input format (16 features)
    
    Feature order matches training data:
    0-5: Spectral bands (B2, B3, B4, B8, B11, B12)
    6-10: Spectral indices (NDVI, NDWI, NDBI, NBR, NDMI)
    11-13: Terrain (elevation, slope, aspect)
    14-15: Weather (temperature, precipitation)
    """
    feature_vector = [
        # Spectral bands (normalized to 0-1 range)
        features.get('B2', {}).get('mean', 0) / 3000.0,
        features.get('B3', {}).get('mean', 0) / 3000.0,
        features.get('B4', {}).get('mean', 0) / 3000.0,
        features.get('B8', {}).get('mean', 0) / 3000.0,
        features.get('B11', {}).get('mean', 0) / 3000.0,
        features.get('B12', {}).get('mean', 0) / 3000.0,
        
        # Spectral indices (already normalized -1 to 1)
        features.get('NDVI', {}).get('mean', 0),
        features.get('NDWI', {}).get('mean', 0),
        features.get('NDBI', {}).get('mean', 0),
        features.get('NBR', {}).get('mean', 0),
        features.get('NDMI', {}).get('mean', 0),
        
        # Terrain features
        features.get('elevation', {}).get('mean', 0) / 1000.0,  # Normalize to 0-8 range
        features.get('slope', {}).get('mean', 0) / 45.0,  # Normalize to 0-2 range
        features.get('aspect', {}).get('mean', 0) / 360.0,  # Normalize to 0-1 range
        
        # Weather features
        (weather.get('temperature_celsius', 15) + 20) / 60.0 if weather.get('temperature_celsius') else 0.5,  # Normalize -20°C to 40°C
        weather.get('precipitation_mm_30day', 50) / 200.0 if weather.get('precipitation_mm_30day') is not None else 0.25,  # Normalize 0-200mm
    ]
    
    return torch.FloatTensor(feature_vector).unsqueeze(0)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'ee_initialized': EE_INITIALIZED,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_location():
    """
    Main endpoint to analyze a location
    
    Request body:
    {
        "lat": float,
        "lng": float,
        "buffer_km": float (optional, default 5)
    }
    """
    try:
        data = request.json
        lat = float(data.get('lat'))
        lng = float(data.get('lng'))
        buffer_km = float(data.get('buffer_km', 5))
        
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return jsonify({'error': 'Invalid coordinates. Latitude must be -90 to 90, longitude -180 to 180'}), 400
        
        if not (0.5 <= buffer_km <= 50):
            return jsonify({'error': 'Buffer must be between 0.5 and 50 km'}), 400
        
        print(f"[{datetime.now().isoformat()}] Analyzing location: {lat}, {lng} (buffer: {buffer_km}km)")
        
        # Get satellite features
        print("  → Fetching satellite imagery...")
        satellite_data = get_satellite_features(lat, lng, buffer_km)
        print(f"  ✓ Retrieved {satellite_data['imagery_count']} images, sampled {satellite_data['sample_size']} points")
        
        # Get weather data
        print("  → Fetching weather data...")
        weather_data = get_weather_data(lat, lng)
        print(f"  ✓ Temperature: {weather_data.get('temperature_fahrenheit')}°F, Precipitation: {weather_data.get('precipitation_mm_30day')}mm")
        
        # Prepare model input
        model_input = prepare_model_input(satellite_data['features'], weather_data)
        
        # Run inference
        print("  → Running model inference...")
        with torch.no_grad():
            logits = MODEL(model_input)
            probabilities = torch.softmax(logits, dim=1)
            prediction = logits.argmax(dim=1).item()
        
        # Extract probabilities
        prob_class_0 = probabilities[0][0].item()  # Low risk
        prob_class_1 = probabilities[0][1].item()  # High risk
        
        # Calculate risk scores based on environmental factors
        ndvi = satellite_data['features'].get('NDVI', {}).get('mean', 0)
        ndwi = satellite_data['features'].get('NDWI', {}).get('mean', 0)
        nbr = satellite_data['features'].get('NBR', {}).get('mean', 0)
        elevation = satellite_data['features'].get('elevation', {}).get('mean', 0)
        slope = satellite_data['features'].get('slope', {}).get('mean', 0)
        temp = weather_data.get('temperature_celsius', 15)
        precip = weather_data.get('precipitation_mm_30day', 50)
        
        # Wildfire risk factors
        wildfire_risk = prob_class_1
        
        # Increase wildfire risk if:
        # - Low vegetation moisture (low NDVI)
        # - Low water content (low NDWI)
        # - High temperature
        # - Low precipitation
        wildfire_multiplier = 1.0
        if ndvi < 0.2:  # Low vegetation
            wildfire_multiplier *= 1.2
        if ndwi < 0.0:  # Dry conditions
            wildfire_multiplier *= 1.3
        if temp and temp > 30:  # Hot (>86°F)
            wildfire_multiplier *= 1.2
        if precip and precip < 20:  # Drought conditions
            wildfire_multiplier *= 1.3
        
        wildfire_risk = min(wildfire_risk * wildfire_multiplier, 0.99)
        
        # Flood risk factors
        flood_risk = prob_class_1 * 0.6  # Base from model
        
        # Increase flood risk if:
        # - Low elevation
        # - Low slope (flat areas)
        # - High precipitation
        # - Near water (high NDWI)
        flood_multiplier = 1.0
        if elevation < 50:  # Low elevation
            flood_multiplier *= 1.4
        if slope < 2:  # Flat terrain
            flood_multiplier *= 1.3
        if precip and precip > 100:  # Heavy rainfall
            flood_multiplier *= 1.4
        if ndwi > 0.3:  # Near water bodies
            flood_multiplier *= 1.2
        
        flood_risk = min(flood_risk * flood_multiplier, 0.99)
        
        print(f"  ✓ Analysis complete - Wildfire: {wildfire_risk:.1%}, Flood: {flood_risk:.1%}")
        
        response = {
            'location': {
                'lat': lat,
                'lng': lng,
                'buffer_km': buffer_km
            },
            'predictions': {
                'wildfire_risk': round(wildfire_risk, 4),
                'flood_risk': round(flood_risk, 4),
                'model_confidence': round(max(prob_class_0, prob_class_1), 4)
            },
            'features': {
                'ndvi': round(satellite_data['features'].get('NDVI', {}).get('mean', 0), 4),
                'ndwi': round(satellite_data['features'].get('NDWI', {}).get('mean', 0), 4),
                'ndbi': round(satellite_data['features'].get('NDBI', {}).get('mean', 0), 4),
                'nbr': round(satellite_data['features'].get('NBR', {}).get('mean', 0), 4),
                'elevation': round(satellite_data['features'].get('elevation', {}).get('mean', 0), 1),
                'slope': round(satellite_data['features'].get('slope', {}).get('mean', 0), 2),
                'temperature': weather_data.get('temperature_fahrenheit'),
                'temperature_celsius': weather_data.get('temperature_celsius'),
                'precipitation': weather_data.get('precipitation_mm_30day')
            },
            'satellite_image': satellite_data['image_url'],
            'metadata': {
                'sample_size': satellite_data['sample_size'],
                'imagery_count': satellite_data['imagery_count'],
                'acquisition_date': satellite_data['acquisition_date'],
                'date_range': satellite_data['date_range'],
                'weather_data_source': weather_data.get('data_source'),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Analysis failed: {error_msg}")
        return jsonify({
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'location': {'lat': lat, 'lng': lng} if 'lat' in locals() else None
        }), 500

@app.before_request
def initialize_services():
    """Initialize services before first request"""
    global MODEL, DEVICE, EE_INITIALIZED
    
    if MODEL is None:
        MODEL, DEVICE = load_model()
    
    if not EE_INITIALIZED:
        EE_INITIALIZED = init_ee()

if __name__ == '__main__':
    print("=" * 60)
    print("Wildfire & Flood Risk Assessment API")
    print("Production-Grade Satellite Analysis System")
    print("=" * 60)
    
    # Initialize on startup
    MODEL, DEVICE = load_model()
    EE_INITIALIZED = init_ee()
    
    print("\nStarting server on port 5000...")
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  POST /analyze - Analyze location")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)