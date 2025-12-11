from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
CORS(app)

def init_ee():
    """Initialize Google Earth Engine with service account"""
    try:
        import os, json, tempfile
        
        # Get credentials from environment variable
        creds_json = os.environ.get('EE_CREDENTIALS')
        if not creds_json:
            print("No EE_CREDENTIALS environment variable found")
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
                creds['client_email'],
                temp_file
            )
            ee.Initialize(credentials)
            print(f"✓ Earth Engine initialized with project: {creds.get('project_id', 'unknown')}")
            return True
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {str(e)[:100]}")
        return False

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
    
    if not os.path.exists(model_path):
        print(f"✗ ERROR: Model not found at {model_path}")
        print(f"Current directory: {os.getcwd()}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = DenseModel(input_dim=16)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    return model, device

# Global instances
MODEL = None
DEVICE = None
EE_INITIALIZED = False

def geocode_address(address):
    """
    Primary geocoder for your backend. Tries OpenStreetMap first,
    falls back to US Census Bureau. Returns lat/lng or None.
    """
    # Normalize the input address
    address = ', '.join([part.strip() for part in address.split(',') if part.strip()])
    print(f"  → Geocoding: '{address}'")

    # --- Attempt 1: OpenStreetMap (Nominatim) ---
    try:
        parts = [p.strip() for p in address.split(',')]
        has_street = False
        if len(parts) > 0:
            first_part = parts[0].lower()
            street_indicators = ['ave', 'st', 'rd', 'ln', 'dr', 'blvd', 'way', 'ct', 'hw', 'route']
            has_street = any(char.isdigit() for char in first_part) or any(ind in first_part for ind in street_indicators)

        url = "https://nominatim.openstreetmap.org/search"
        headers = {
            'User-Agent': 'WildfireFloodRiskAssessment/1.0 (contact: rfarrales25@gmail.com)'  # Your email
        }

        if has_street and len(parts) >= 2:
            params = {
                'street': parts[0],
                'city': parts[1] if len(parts) > 1 else '',
                'state': parts[2] if len(parts) > 2 else '',
                'country': 'United States',
                'format': 'jsonv2',
                'limit': 1,
                'countrycodes': 'us'
            }
        else:
            search_query = address if address.lower().endswith('usa') else f"{address}, USA"
            params = {
                'q': search_query,
                'format': 'jsonv2',
                'limit': 1,
                'countrycodes': 'us'
            }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            result = data[0]
            print(f"  ✓ OpenStreetMap succeeded.")
            return {
                'lat': float(result['lat']),
                'lng': float(result['lon']),
                'display_name': result.get('display_name', address),
                'type': result.get('type', 'unknown')
            }
    except Exception as e:
        print(f"    OSM attempt failed: {e}")

    # --- Attempt 2: US Census Bureau Geocoder (Fallback) ---
    print(f"  ⚠ OpenStreetMap failed. Falling back to US Census Geocoder...")
    try:
        url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
        params = {
            'address': address,
            'benchmark': 'Public_AR_Current',
            'format': 'json'
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get('result', {}).get('addressMatches'):
            match = data['result']['addressMatches'][0]
            coords = match['coordinates']
            print(f"  ✓ US Census Geocoder succeeded.")
            return {
                'lat': float(coords['y']),  # Census uses y=lat, x=lng
                'lng': float(coords['x']),
                'display_name': match.get('matchedAddress', address),
                'type': 'census'
            }
    except Exception as e:
        print(f"    Census geocoder also failed: {e}")

    # --- All attempts failed ---
    print(f"  ✗ All geocoding attempts failed for: {address}")
    return None

def get_historical_trends(lat, lng):
    """
    Analyze historical trends for the location using multi-year satellite data
    Returns 5-year trend analysis
    """
    try:
        point = ee.Geometry.Point([lng, lat])
        region = point.buffer(5000)
        
        current_year = datetime.now().year
        trends = []
        
        # Get data for past 5 years
        for year_offset in range(5, 0, -1):
            year = current_year - year_offset
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(region) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            if s2.size().getInfo() > 0:
                image = s2.median()
                ndvi = image.normalizedDifference(['B8', 'B4'])
                ndwi = image.normalizedDifference(['B3', 'B8'])
                
                sample = ndvi.addBands(ndwi).sample(
                    region=region, 
                    scale=100, 
                    numPixels=20
                ).getInfo()
                
                if sample['features']:
                    ndvi_vals = [f['properties'].get('nd', 0) for f in sample['features']]
                    ndwi_vals = [f['properties'].get('nd_1', 0) for f in sample['features']]
                    
                    trends.append({
                        'year': year,
                        'vegetation_index': round(np.mean(ndvi_vals), 3),
                        'water_index': round(np.mean(ndwi_vals), 3)
                    })
        
        if len(trends) >= 3:
            veg_trend = trends[-1]['vegetation_index'] - trends[0]['vegetation_index']
            water_trend = trends[-1]['water_index'] - trends[0]['water_index']
            
            return {
                'historical_data': trends,
                'vegetation_trend': 'decreasing' if veg_trend < -0.05 else 'increasing' if veg_trend > 0.05 else 'stable',
                'water_trend': 'decreasing' if water_trend < -0.05 else 'increasing' if water_trend > 0.05 else 'stable',
                'vegetation_change': round(veg_trend, 3),
                'water_change': round(water_trend, 3),
                'years_analyzed': len(trends)
            }
        else:
            return None
            
    except Exception as e:
        print(f"Historical analysis failed: {e}")
        return None

def get_weather_data(lat, lng):
    """Get actual temperature and precipitation data from ERA5"""
    try:
        point = ee.Geometry.Point([lng, lat])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # ERA5 Temperature
        era5_temp = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filterBounds(point) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .select('temperature_2m')
        
        # ERA5 Precipitation
        era5_precip = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filterBounds(point) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .select('total_precipitation_sum')
        
        temp_fahrenheit = None
        temp_celsius = None
        if era5_temp.size().getInfo() > 0:
            temp_mean = era5_temp.mean()
            temp_sample = temp_mean.sample(region=point, scale=11132, numPixels=1).first().getInfo()
            temp_kelvin = temp_sample['properties'].get('temperature_2m', 288.15)
            temp_celsius = temp_kelvin - 273.15
            temp_fahrenheit = (temp_celsius * 9/5) + 32
        
        precip_mm = None
        if era5_precip.size().getInfo() > 0:
            precip_sum = era5_precip.sum()
            precip_sample = precip_sum.sample(region=point, scale=11132, numPixels=1).first().getInfo()
            precip_meters = precip_sample['properties'].get('total_precipitation_sum', 0)
            precip_mm = precip_meters * 1000
        
        return {
            'temperature_fahrenheit': round(temp_fahrenheit, 1) if temp_fahrenheit else None,
            'temperature_celsius': round(temp_celsius, 1) if temp_celsius else None,
            'precipitation_mm_30day': round(precip_mm, 1) if precip_mm is not None else None,
            'data_source': 'ERA5_LAND',
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
    """Fetch satellite imagery and compute features"""
    if not EE_INITIALIZED:
        raise Exception("Earth Engine not initialized")
    
    point = ee.Geometry.Point([lng, lat])
    region = point.buffer(buffer_km * 1000)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Get Sentinel-2 imagery
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    
    collection_size = s2.size().getInfo()
    if collection_size == 0:
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        
        collection_size = s2.size().getInfo()
        if collection_size == 0:
            raise Exception("No Sentinel-2 imagery available for this location in the past 90 days")
    
    image = s2.median()
    
    # Compute spectral indices
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    # Terrain data
    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.select('elevation')
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    
    combined = image.addBands([ndvi, ndwi, ndbi, nbr, ndmi, elevation, slope, aspect])
    
    # Sample features
    sample = combined.sample(
        region=region, 
        scale=30,
        numPixels=100,
        seed=42,
        geometries=True
    ).getInfo()
    
    if not sample['features'] or len(sample['features']) == 0:
        raise Exception("Could not extract features from imagery - area may be over water or outside coverage")
    
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
            feature_dict[band] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    # Generate satellite image visualization
    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'gamma': 1.4
    }
    
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
    
    # Get acquisition date
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

def identify_danger_factors(satellite_data, weather_data, wildfire_risk, flood_risk):
    """
    Identify and rank the most prevalent danger factors with scientific citations
    """
    factors = []
    
    # Extract metrics
    ndvi = satellite_data['features'].get('NDVI', {}).get('mean', 0)
    ndwi = satellite_data['features'].get('NDWI', {}).get('mean', 0)
    nbr = satellite_data['features'].get('NBR', {}).get('mean', 0)
    ndmi = satellite_data['features'].get('NDMI', {}).get('mean', 0)
    elevation = satellite_data['features'].get('elevation', {}).get('mean', 0)
    slope = satellite_data['features'].get('slope', {}).get('mean', 0)
    temp_c = weather_data.get('temperature_celsius')
    precip = weather_data.get('precipitation_mm_30day')
    
    # WILDFIRE FACTORS
    if ndvi < 0.3:
        severity = 'high' if ndvi < 0.15 else 'moderate'
        factors.append({
            'type': 'wildfire',
            'factor': 'Low Vegetation Density',
            'severity': severity,
            'value': f"NDVI: {ndvi:.3f}",
            'description': 'Sparse or stressed vegetation increases fire spread potential and reduces natural firebreaks.',
            'citation': 'Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. Remote Sensing of Environment, 8(2), 127-150.',
            'explanation': f'Normal vegetation shows NDVI > 0.4. This area shows {ndvi:.3f}, indicating sparse cover.'
        })
    
    if ndwi < 0.0:
        severity = 'high' if ndwi < -0.2 else 'moderate'
        factors.append({
            'type': 'wildfire',
            'factor': 'Drought Conditions',
            'severity': severity,
            'value': f"NDWI: {ndwi:.3f}",
            'description': 'Low water content in vegetation and soil creates ideal conditions for fire ignition and rapid spread.',
            'citation': 'McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing, 17(7), 1425-1432.',
            'explanation': f'Negative NDWI indicates very dry conditions. This area shows {ndwi:.3f}.'
        })
    
    if ndmi < 0.2:
        severity = 'high' if ndmi < 0.0 else 'moderate'
        factors.append({
            'type': 'wildfire',
            'factor': 'Low Fuel Moisture Content',
            'severity': severity,
            'value': f"NDMI: {ndmi:.3f}",
            'description': 'Dry vegetation acts as highly combustible fuel material, significantly increasing fire intensity.',
            'citation': 'Wilson, E.H., & Sader, S.A. (2002). Detection of forest harvest type using multiple dates of Landsat TM imagery. Remote Sensing of Environment, 80(3), 385-396.',
            'explanation': f'Healthy vegetation shows NDMI > 0.4. This area shows {ndmi:.3f}, indicating dry fuels.'
        })
    
    if temp_c and temp_c > 30:
        severity = 'high' if temp_c > 35 else 'moderate'
        factors.append({
            'type': 'wildfire',
            'factor': 'Elevated Temperature',
            'severity': severity,
            'value': f"{temp_c:.1f}°C ({temp_c * 9/5 + 32:.1f}°F)",
            'description': 'High temperatures dry out fuel sources and increase the likelihood of fire ignition.',
            'citation': 'Abatzoglou, J.T., & Williams, A.P. (2016). Impact of anthropogenic climate change on wildfire across western US forests. PNAS, 113(42), 11770-11775.',
            'explanation': f'Fire danger increases significantly above 30°C. Current temperature: {temp_c:.1f}°C.'
        })
    
    if precip and precip < 20:
        severity = 'high' if precip < 10 else 'moderate'
        factors.append({
            'type': 'wildfire',
            'factor': 'Precipitation Deficit',
            'severity': severity,
            'value': f"{precip:.1f}mm (30-day total)",
            'description': 'Prolonged low rainfall creates extended dry periods that are highly favorable for fire ignition and spread.',
            'citation': 'Littell, J.S., et al. (2009). Climate and wildfire area burned in western U.S. ecoprovinces, 1916-2003. Ecological Applications, 19(4), 1003-1021.',
            'explanation': f'Normal monthly precipitation: 50-100mm. This area received only {precip:.1f}mm.'
        })
    
    # FLOOD FACTORS
    if elevation < 50:
        severity = 'high' if elevation < 10 else 'moderate'
        factors.append({
            'type': 'flood',
            'factor': 'Low Elevation',
            'severity': severity,
            'value': f"{elevation:.1f}m above sea level",
            'description': 'Low-lying areas are more susceptible to water accumulation and have limited natural drainage.',
            'citation': 'FEMA (2020). Flood Insurance Study Guidelines and Specifications for Study Contractors. Federal Emergency Management Agency.',
            'explanation': f'Areas below 50m elevation face higher flood risk. Current elevation: {elevation:.1f}m.'
        })
    
    if slope < 2:
        severity = 'high' if slope < 0.5 else 'moderate'
        factors.append({
            'type': 'flood',
            'factor': 'Flat Terrain',
            'severity': severity,
            'value': f"{slope:.2f}° slope",
            'description': 'Flat areas have poor natural drainage, leading to water pooling and extended inundation periods.',
            'citation': 'Jha, A.K., et al. (2012). Cities and Flooding: A Guide to Integrated Urban Flood Risk Management for the 21st Century. World Bank.',
            'explanation': f'Slopes below 2° have poor drainage. This area shows {slope:.2f}° slope.'
        })
    
    if ndwi > 0.3:
        severity = 'high' if ndwi > 0.5 else 'moderate'
        factors.append({
            'type': 'flood',
            'factor': 'Proximity to Water Bodies',
            'severity': severity,
            'value': f"NDWI: {ndwi:.3f}",
            'description': 'Areas near rivers, lakes, or wetlands face higher flood risk during heavy precipitation events.',
            'citation': 'Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features. International Journal of Remote Sensing, 27(14), 3025-3033.',
            'explanation': f'NDWI > 0.3 indicates proximity to water. This area shows {ndwi:.3f}.'
        })
    
    if precip and precip > 100:
        severity = 'high' if precip > 150 else 'moderate'
        factors.append({
            'type': 'flood',
            'factor': 'Heavy Precipitation',
            'severity': severity,
            'value': f"{precip:.1f}mm (30-day total)",
            'description': 'Excessive rainfall saturates soil and overwhelms drainage systems, leading to flash flooding.',
            'citation': 'Kunkel, K.E., et al. (2013). Monitoring and Understanding Trends in Extreme Storms. Bulletin of the American Meteorological Society, 94(4), 499-514.',
            'explanation': f'Monthly precipitation > 100mm increases flood risk. This area received {precip:.1f}mm.'
        })
    
    # Sort by severity
    wildfire_factors = [f for f in factors if f['type'] == 'wildfire']
    flood_factors = [f for f in factors if f['type'] == 'flood']
    
    severity_order = {'high': 3, 'moderate': 2, 'low': 1}
    wildfire_factors.sort(key=lambda x: severity_order[x['severity']], reverse=True)
    flood_factors.sort(key=lambda x: severity_order[x['severity']], reverse=True)
    
    return {
        'wildfire_factors': wildfire_factors,
        'flood_factors': flood_factors,
        'total_factors': len(factors)
    }

def prepare_model_input(features, weather):
    """Convert features to model input format (16 features)"""
    feature_vector = [
        features.get('B2', {}).get('mean', 0) / 3000.0,
        features.get('B3', {}).get('mean', 0) / 3000.0,
        features.get('B4', {}).get('mean', 0) / 3000.0,
        features.get('B8', {}).get('mean', 0) / 3000.0,
        features.get('B11', {}).get('mean', 0) / 3000.0,
        features.get('B12', {}).get('mean', 0) / 3000.0,
        features.get('NDVI', {}).get('mean', 0),
        features.get('NDWI', {}).get('mean', 0),
        features.get('NDBI', {}).get('mean', 0),
        features.get('NBR', {}).get('mean', 0),
        features.get('NDMI', {}).get('mean', 0),
        features.get('elevation', {}).get('mean', 0) / 1000.0,
        features.get('slope', {}).get('mean', 0) / 45.0,
        features.get('aspect', {}).get('mean', 0) / 360.0,
        (weather.get('temperature_celsius', 15) + 20) / 60.0 if weather.get('temperature_celsius') else 0.5,
        weather.get('precipitation_mm_30day', 50) / 200.0 if weather.get('precipitation_mm_30day') is not None else 0.25,
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
        'version': '3.0.0'
    })

@app.route('/geocode', methods=['POST'])
def geocode():
    """Convert address to coordinates"""
    try:
        data = request.json
        address = data.get('address')
        
        if not address:
            return jsonify({'error': 'Address is required'}), 400
        
        result = geocode_address(address)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Location not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_location():
    """
    Comprehensive risk analysis endpoint
    Accepts: {"lat": float, "lng": float} OR {"address": string}
    """
    try:
        data = request.json
        
        # Handle address or coordinates
        if 'address' in data:
            print(f"[{datetime.now().isoformat()}] Geocoding address: {data['address']}")
            geo_result = geocode_address(data['address'])
            if not geo_result:
                return jsonify({'error': 'Could not geocode address. Please check spelling and try again.'}), 400
            lat = geo_result['lat']
            lng = geo_result['lng']
            location_name = geo_result['display_name']
        else:
            lat = float(data.get('lat'))
            lng = float(data.get('lng'))
            location_name = f"{lat:.4f}°, {lng:.4f}°"
        
        buffer_km = float(data.get('buffer_km', 5))
        
        # Validate
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return jsonify({'error': 'Invalid coordinates. Latitude must be -90 to 90, longitude -180 to 180.'}), 400
        
        if not (0.5 <= buffer_km <= 50):
            return jsonify({'error': 'Buffer must be between 0.5 and 50 km'}), 400
        
        print(f"[{datetime.now().isoformat()}] Analyzing: {location_name}")
        
        # Get satellite features and imagery
        print("  → Fetching satellite imagery...")
        satellite_data = get_satellite_features(lat, lng, buffer_km)
        print(f"  ✓ Retrieved {satellite_data['imagery_count']} images, sampled {satellite_data['sample_size']} points")
        
        # Get weather data
        print("  → Fetching weather data...")
        weather_data = get_weather_data(lat, lng)
        print(f"  ✓ Temperature: {weather_data.get('temperature_fahrenheit')}°F, Precipitation: {weather_data.get('precipitation_mm_30day')}mm")
        
        # Get historical trends
        print("  → Analyzing historical trends...")
        historical = get_historical_trends(lat, lng)
        if historical:
            print(f"  ✓ Historical: Vegetation {historical['vegetation_trend']}, Water {historical['water_trend']}")
        else:
            print("  ⚠ Insufficient historical data")
        
        # Run model inference
        model_input = prepare_model_input(satellite_data['features'], weather_data)
        
        print("  → Running AI model...")
        with torch.no_grad():
            logits = MODEL(model_input)
            probabilities = torch.softmax(logits, dim=1)
        
        prob_class_0 = probabilities[0][0].item()
        prob_class_1 = probabilities[0][1].item()
        
        # Calculate risk scores with environmental multipliers
        ndvi = satellite_data['features'].get('NDVI', {}).get('mean', 0)
        ndwi = satellite_data['features'].get('NDWI', {}).get('mean', 0)
        elevation = satellite_data['features'].get('elevation', {}).get('mean', 0)
        slope = satellite_data['features'].get('slope', {}).get('mean', 0)
        temp = weather_data.get('temperature_celsius', 15)
        precip = weather_data.get('precipitation_mm_30day', 50)
        
        # Wildfire risk
        wildfire_risk = prob_class_1
        wildfire_multiplier = 1.0
        if ndvi < 0.2: wildfire_multiplier *= 1.2
        if ndwi < 0.0: wildfire_multiplier *= 1.3
        if temp and temp > 30: wildfire_multiplier *= 1.2
        if precip and precip < 20: wildfire_multiplier *= 1.3
        wildfire_risk = min(wildfire_risk * wildfire_multiplier, 0.99)
        
        # Flood risk
        flood_risk = prob_class_1 * 0.6
        flood_multiplier = 1.0
        if elevation < 50: flood_multiplier *= 1.4
        if slope < 2: flood_multiplier *= 1.3
        if precip and precip > 100: flood_multiplier *= 1.4
        if ndwi > 0.3: flood_multiplier *= 1.2
        flood_risk = min(flood_risk * flood_multiplier, 0.99)
        
        print(f"  ✓ Risk calculated - Wildfire: {wildfire_risk:.1%}, Flood: {flood_risk:.1%}")
        
        # Identify danger factors with citations
        print("  → Identifying danger factors...")
        danger_analysis = identify_danger_factors(satellite_data, weather_data, wildfire_risk, flood_risk)
        print(f"  ✓ Found {danger_analysis['total_factors']} risk factors")
        
        # Build comprehensive response
        response = {
            'location': {
                'lat': lat,
                'lng': lng,
                'name': location_name,
                'buffer_km': buffer_km
            },
            'predictions': {
                'wildfire_risk': round(wildfire_risk, 4),
                'flood_risk': round(flood_risk, 4),
                'model_confidence': round(max(prob_class_0, prob_class_1), 4)
            },
            'satellite_image': satellite_data['image_url'],
            'features': {
                'ndvi': round(satellite_data['features'].get('NDVI', {}).get('mean', 0), 4),
                'ndwi': round(satellite_data['features'].get('NDWI', {}).get('mean', 0), 4),
                'ndbi': round(satellite_data['features'].get('NDBI', {}).get('mean', 0), 4),
                'nbr': round(satellite_data['features'].get('NBR', {}).get('mean', 0), 4),
                'ndmi': round(satellite_data['features'].get('NDMI', {}).get('mean', 0), 4),
                'elevation': round(satellite_data['features'].get('elevation', {}).get('mean', 0), 1),
                'slope': round(satellite_data['features'].get('slope', {}).get('mean', 0), 2),
                'aspect': round(satellite_data['features'].get('aspect', {}).get('mean', 0), 1),
                'temperature': weather_data.get('temperature_fahrenheit'),
                'temperature_celsius': weather_data.get('temperature_celsius'),
                'precipitation': weather_data.get('precipitation_mm_30day')
            },
            'danger_factors': danger_analysis,
            'historical_trends': historical if historical else {
                'message': 'Insufficient historical data available for this location',
                'years_analyzed': 0
            },
            'metadata': {
                'sample_size': satellite_data['sample_size'],
                'imagery_count': satellite_data['imagery_count'],
                'acquisition_date': satellite_data['acquisition_date'],
                'date_range': satellite_data['date_range'],
                'weather_data_source': weather_data.get('data_source'),
                'timestamp': datetime.now().isoformat(),
                'analysis_version': '3.0.0'
            }
        }
        
        print(f"  ✓ Analysis complete!")
        print("=" * 60)
        return jsonify(response)
    
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Analysis failed: {error_msg}")
        return jsonify({
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'location': {'lat': lat, 'lng': lng, 'name': location_name} if 'lat' in locals() else None
        }), 500

@app.before_request
def initialize_services():
    """Initialize services before first request"""
    global MODEL, DEVICE, EE_INITIALIZED
    
    if MODEL is None:
        try:
            MODEL, DEVICE = load_model()
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    if not EE_INITIALIZED:
        EE_INITIALIZED = init_ee()

if __name__ == '__main__':
    print("=" * 70)
    print(" " * 15 + "WILDFIRE & FLOOD RISK ASSESSMENT API v3.0")
    print(" " * 15 + "Production-Grade Satellite Analysis System")
    print("=" * 70)
    print()
    print("FEATURES:")
    print("  ✓ Address-to-Coordinate Geocoding")
    print("  ✓ Live Sentinel-2 Satellite Imagery")
    print("  ✓ 16-Feature AI Risk Assessment")
    print("  ✓ Danger Factor Identification with Citations")
    print("  ✓ 5-Year Historical Trend Analysis")
    print("  ✓ Real Weather Data (ERA5-Land)")
    print()
    print("DATA SOURCES:")
    print("  • Sentinel-2 MSI (10m resolution)")
    print("  • ERA5-Land Climate Reanalysis")
    print("  • SRTM Elevation Data")
    print("  • OpenStreetMap Geocoding")
    print()
    print("=" * 70)
    
    # Initialize on startup
    print("\nInitializing services...")
    try:
        MODEL, DEVICE = load_model()
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("  Please ensure model file is in: satellite_risk_assessment_0437/models/")
        
    EE_INITIALIZED = init_ee()
    
    if not EE_INITIALIZED:
        print("✗ WARNING: Earth Engine not initialized. Set EE_CREDENTIALS environment variable.")
    
    print("\n" + "=" * 70)
    print("STARTING SERVER...")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  /health           - System health check")
    print("  POST /geocode          - Convert address to coordinates")
    print("  POST /analyze          - Comprehensive risk analysis")
    print("\nServer running on: http://0.0.0.0:5000")
    print("Press CTRL+C to stop")
    print("=" * 70)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
