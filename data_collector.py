import requests
import json
import pandas as pd
from config import API_KEY

class SmithsonianDataCollector:
    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.si.edu/openaccess/api/v1.0/search"
    
    def search_collections(self, query, rows=100, start=0):
        """
        Search the Smithsonian collections with the given query
        """
        params = {
            "api_key": self.api_key,
            "q": query,
            "rows": rows,
            "start": start
        }
        
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def convert_to_dataframe(self, data):
        """
        Convert API response to a pandas DataFrame
        """
        rows = []
        
        for item in data['response']['rows']:
            # Extract the core metadata
            row = {
                'id': item.get('id', ''),
                'title': item.get('title', ''),
                'type': item.get('type', ''),
                'url': item.get('content', {}).get('descriptiveNonRepeating', {}).get('record_link', ''),
                'date': item.get('content', {}).get('indexedStructured', {}).get('date', ''),
                'topic': str(item.get('content', {}).get('indexedStructured', {}).get('topic', [])),
                'object_type': str(item.get('content', {}).get('indexedStructured', {}).get('object_type', [])),
                'online_media_type': str(item.get('content', {}).get('descriptiveNonRepeating', {}).get('online_media', {}).get('mediaType', [])),
                'data_source': item.get('content', {}).get('descriptiveNonRepeating', {}).get('data_source', ''),
            }
            
            # Extract creator/contributor information
            creators = item.get('content', {}).get('indexedStructured', {}).get('name', [])
            row['creators'] = str(creators) if creators else ''
            
            # Get thumbnail image if available
            media = item.get('content', {}).get('descriptiveNonRepeating', {}).get('online_media', {}).get('media', [])
            if media and len(media) > 0:
                row['thumbnail'] = media[0].get('thumbnail', '')
            else:
                row['thumbnail'] = ''
                
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def transform_to_json_ld(self, df):
        """
        Transform a DataFrame to JSON-LD format
        """
        context = {
            "@context": {
                "@vocab": "http://schema.org/",
                "si": "https://www.si.edu/terms/",
                "id": "@id",
                "type": "@type",
                "creator": "creator",
                "title": "name",
                "topic": "about",
                "date": "dateCreated",
                "object_type": "si:objectType",
                "thumbnail": "image",
                "url": "url",
                "data_source": "provider"
            }
        }
        
        items = []
        
        for _, row in df.iterrows():
            item = {
                "@id": f"https://si.edu/object/{row['id']}",
                "@type": "CreativeWork",
                "name": row['title'],
                "provider": row['data_source'],
                "si:objectType": row['object_type'],
                "about": row['topic']
            }
            
            # Add non-empty values
            if row['date']:
                item["dateCreated"] = row['date']
            
            if row['url']:
                item["url"] = row['url']
                
            if row['thumbnail']:
                item["image"] = row['thumbnail']
                
            if row['creators']:
                # Try to parse the string representation of a list back to a list
                try:
                    creators_list = eval(row['creators'])
                    if isinstance(creators_list, list):
                        item["creator"] = creators_list
                    else:
                        item["creator"] = row['creators']
                except:
                    item["creator"] = row['creators']
            
            items.append(item)
        
        # Combine context with items
        json_ld = context.copy()
        json_ld["@graph"] = items
        
        return json_ld