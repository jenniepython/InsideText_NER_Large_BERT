#!/usr/bin/env python3
"""
Improved Streamlit Entity Linker Application with the bert-large-NER model

A web interface for entity extraction and linking using the bert-large-NER model.
This application focuses purely on BERT NER with linking and geocoding.

Author: Jennie Williams
"""

import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="From Text to Linked Data using Open Source Model: dslim/bert-large-NER",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Authentication is REQUIRED
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        st.error("Authentication required: config.yaml file not found!")
        st.info("Please ensure config.yaml is in the same directory as this app.")
        st.stop()
    
    # Load configuration
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Setup authentication
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Check if already authenticated via session state
    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
        # Continue to app below...
    else:
        # Render login form
        try:
            # Try different login methods
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            # Handle the result
            if login_result is None:
                # Check session state for authentication result
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                        st.info("Try username: demo_user with your password")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()  # Refresh to show authenticated state
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                # Store in session state
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()  # Refresh to show authenticated state
                elif auth_status == False:
                    st.error("Username/password is incorrect")
                    st.stop()
            else:
                st.error(f"Unexpected login result format: {login_result}")
                st.stop()
                
        except Exception as login_error:
            st.error(f"Login method error: {login_error}")
            st.stop()
        
except ImportError:
    st.error("Authentication required: streamlit-authenticator not installed!")
    st.info("Please install streamlit-authenticator to access this application.")
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.info("Cannot proceed without proper authentication.")
    st.stop()

import sys
import subprocess

# Install packages if not available
try:
    import torch
    import transformers
    import numpy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "numpy"])
    import torch
    import transformers
    import numpy

import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from typing import List, Dict, Any, Optional
import sys
import os
import re
import time
import requests
import urllib.parse
import pycountry
import hashlib

class ImprovedBERTEntityLinker:
    """
    Improved BERT-based entity linking class.
    
    This class handles the complete pipeline from text processing to entity
    extraction using the dslim/bert-large-NER model without confidence filtering,
    plus validation, linking, and output generation.
    """
    
    def __init__(self):
        """Initialise the ImprovedBERTEntityLinker."""
        
        # Color scheme for different entity types in HTML output
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORG': '#9fd2cd',             # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOC': '#EFCA89',             # F&B Yellow ground
            'MISC': '#DDD3C0',            # F&B Old White
        }
        
        # Initialise model
        self.ner_pipeline = None
        self._load_model()

    def _load_model(self):
        """Load BERT NER model with optimal settings."""
        try:
            from transformers import pipeline
            
            with st.spinner("Loading BERT-large-NER model..."):
                try:
                    ner_model_name = "dslim/bert-large-NER"
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=ner_model_name,
                        tokenizer=ner_model_name,
                        aggregation_strategy="simple",  # Changed from "max" to "simple"
                        ignore_labels=[]  # Don't ignore any labels
                    )
                    st.success("BERT-large-NER model loaded successfully")
                except Exception as e:
                    st.error(f"Failed to load NER model: {e}")
                    st.stop()
                    
        except ImportError:
            st.error("Required packages not installed. Please install:")
            st.code("pip install transformers torch")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    def extract_entities(self, text: str):
        """Extract named entities using BERT NER model without confidence filtering."""
        entities = []
        
        if not self.ner_pipeline:
            st.error("NER model not loaded")
            return entities
        
        try:
            # Get raw predictions from BERT model
            raw_entities = self.ner_pipeline(text)
            
            # Debug output
            print(f"DEBUG - Found {len(raw_entities)} raw entities:")
            for ent in raw_entities:
                print(f"  '{ent.get('word', 'N/A')}' ({ent.get('entity_group', ent.get('entity', 'N/A'))}) - score: {ent.get('score', 0):.3f}")
            
            # Process ALL entities without confidence filtering
            for ent in raw_entities:
                # Handle different response formats
                entity_type = ent.get('entity_group', ent.get('entity', 'MISC'))
                entity_text = ent.get('word', '')
                confidence = ent.get('score', 0)
                start_pos = ent.get('start', 0)
                end_pos = ent.get('end', len(entity_text))
                
                # Clean entity text - remove tokenizer artifacts
                entity_text = entity_text.replace('##', '').strip()
                
                # Skip very short or empty entities
                if len(entity_text) <= 1:
                    continue
                
                # Skip common false positives
                if entity_text.lower() in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}:
                    continue
                
                # Create entity dictionary
                entity = {
                    'text': entity_text,
                    'type': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'confidence': confidence,
                    'extraction_method': 'bert_ner'
                }
                
                # Add contextual information
                entity['sentence_context'] = self._extract_sentence_context(entity, text)
                entity['context_keywords'] = self._extract_context_keywords(entity_text, text)
                entity['entity_frequency'] = text.lower().count(entity_text.lower())
                
                entities.append(entity)
                
        except Exception as e:
            st.error(f"Entity extraction failed: {e}")
            print(f"DEBUG - Exception in entity extraction: {e}")
        
        # Remove overlapping entities (keep highest confidence)
        entities = self._remove_overlapping_entities(entities)
        
        return entities

    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the highest confidence ones."""
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:
                # Check if entities overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # Keep the higher confidence entity
                    if entity.get('confidence', 0) > existing.get('confidence', 0):
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered

    def _extract_sentence_context(self, entity: Dict[str, Any], text: str) -> str:
        """Extract the sentence containing the entity."""
        # Find sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if entity['text'] in sentence:
                return sentence.strip()
        
        # Fallback: extract context window around entity
        start = max(0, entity['start'] - 100)
        end = min(len(text), entity['end'] + 100)
        return text[start:end].strip()

    def _extract_context_keywords(self, entity_text: str, text: str) -> List[str]:
        """Extract relevant keywords from the context around an entity."""
        entity_pos = text.lower().find(entity_text.lower())
        if entity_pos == -1:
            return []
        
        # Extract surrounding text
        start = max(0, entity_pos - 100)
        end = min(len(text), entity_pos + len(entity_text) + 100)
        context = text[start:end]
        
        # Extract meaningful words (capitalized words and important terms)
        keywords = []
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', context)
        keywords.extend(words)
        
        # Add some common important terms
        important_terms = ['theatre', 'stage', 'company', 'organization', 'university', 'hospital', 
                          'government', 'technology', 'research', 'development', 'market', 'industry']
        for term in important_terms:
            if term in context.lower():
                keywords.append(term)
        
        return list(set(keywords))[:5]  # Return top 5 unique keywords

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Detect geographical context from the text using pycountry for countries."""
        context_clues = []
        text_lower = text.lower()
        
        # Create a lookup set of country names and common variations
        country_names = {}
        try:
            for country in pycountry.countries:
                names = {country.name.lower()}
                if hasattr(country, 'official_name'):
                    names.add(country.official_name.lower())
                names.add(country.alpha_2.lower())
                names.add(country.alpha_3.lower())
                
                for name in names:
                    country_names[name] = country.name.lower()
        except:
            # Fallback to manual list if pycountry fails
            country_names = {
                'uk': 'united kingdom',
                'usa': 'united states',
                'us': 'united states',
                'england': 'united kingdom',
                'france': 'france',
                'germany': 'germany'
            }

        # Check if any known country names are in the text
        for variant, canonical in country_names.items():
            if f" {variant} " in f" {text_lower} " and canonical not in context_clues:
                context_clues.append(canonical)

        # Add GPE/LOC entities if they match a known country
        for entity in entities:
            if entity['type'] in ['GPE', 'LOC']:
                entity_lower = entity['text'].strip().lower()
                if entity_lower in country_names:
                    canonical = country_names[entity_lower]
                    if canonical not in context_clues:
                        context_clues.append(canonical)

        return context_clues[:3]

    def get_coordinates(self, entities):
        """Enhanced coordinate lookup with geographical context detection."""
        context_clues = self._detect_geographical_context(
            st.session_state.get('processed_text', ''), 
            entities
        )
        
        place_types = ['GPE', 'LOC', 'MISC']  # Include MISC as it might contain places
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates
                if entity.get('latitude') is not None:
                    continue
                
                # Try geocoding with context
                if self._try_contextual_geocoding(entity, context_clues):
                    continue
                    
                # Fall back to direct geocoding
                if self._try_direct_geocoding(entity):
                    continue
        
        return entities
    
    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with geographical context."""
        if not context_clues:
            return False

        search_variations = [entity['text']]

        # Add context variations
        for context in context_clues:
            search_variations.append(f"{entity['text']}, {context}")

        # Remove duplicates while preserving order
        search_variations = list(dict.fromkeys(search_variations))

        # Try OpenStreetMap with top 3 context variations
        for search_term in search_variations[:3]:
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': search_term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}

                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = 'openstreetmap_contextual'
                        entity['search_term_used'] = search_term
                        return True

                time.sleep(0.3)  # Be kind to the API
            except Exception:
                continue

        return False
    
    def _try_direct_geocoding(self, entity):
        """Try direct geocoding without context."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': entity['text'],
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'EntityLinker/1.0'}
        
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lon'])
                    entity['location_name'] = result['display_name']
                    entity['geocoding_source'] = 'openstreetmap'
                    return True
        
            time.sleep(0.3)  # Rate limiting
        except Exception:
            pass
        
        return False

    def link_to_wikidata(self, entities):
        """Add Wikidata linking for all entities."""
        for entity in entities:
            try:
                url = "https://www.wikidata.org/w/api.php"

                # Construct search terms
                search_terms = [entity['text']]
                if entity['type'] == 'PERSON':
                    search_terms.append(f"{entity['text']} person")
                elif entity['type'] == 'ORG':
                    search_terms.append(f"{entity['text']} organization")
                    search_terms.append(f"{entity['text']} company")
                elif entity['type'] in ['LOC', 'GPE']:
                    search_terms.append(f"{entity['text']} place")
                    search_terms.append(f"{entity['text']} location")

                if entity.get('context_keywords'):
                    for keyword in entity['context_keywords'][:2]:
                        search_terms.append(f"{entity['text']} {keyword}")

                # Try up to 3 variations
                for search_term in search_terms[:3]:
                    params = {
                        'action': 'wbsearchentities',
                        'format': 'json',
                        'search': search_term,
                        'language': 'en',
                        'limit': 1,
                        'type': 'item'
                    }

                    response = requests.get(url, params=params, timeout=5)
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    if not data.get('search'):
                        continue

                    result = data['search'][0]
                    entity_qid = result['id']

                    # Add Wikidata information
                    entity['wikidata_url'] = f"http://www.wikidata.org/entity/{entity_qid}"
                    entity['wikidata_description'] = result.get('description', '')
                    entity['wikidata_label'] = result.get('label', '')
                    break

                time.sleep(0.1)  # Rate limiting

            except Exception:
                continue

        return entities

    def link_to_wikipedia(self, entities):
        """Add Wikipedia linking for entities without Wikidata links."""
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Use Wikipedia's search API
                search_url = "https://en.wikipedia.org/w/api.php"
                
                # Create enhanced search terms
                search_terms = [entity['text']]
                
                # Add context keywords if available
                if entity.get('context_keywords'):
                    for keyword in entity['context_keywords'][:2]:
                        search_terms.append(f"{entity['text']} {keyword}")
                
                # Try each search term
                for search_term in search_terms[:3]:
                    search_params = {
                        'action': 'query',
                        'format': 'json',
                        'list': 'search',
                        'srsearch': search_term,
                        'srlimit': 1
                    }
                    
                    headers = {'User-Agent': 'EntityLinker/1.0'}
                    response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('query', {}).get('search'):
                            result = data['query']['search'][0]
                            page_title = result['title']
                            
                            # Create Wikipedia URL
                            encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                            entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                            entity['wikipedia_title'] = page_title
                            
                            # Get a snippet/description
                            if result.get('snippet'):
                                snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                                entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                            
                            break  # Found a match, stop searching
                
                time.sleep(0.2)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def enhance_entities_with_context(self, entities, text):
        """Enhance entities with additional contextual information."""
        enhanced_entities = []
        
        for entity in entities:
            # Create a copy of the entity
            enhanced_entity = entity.copy()
            
            # Add nearby entities for relationship context
            enhanced_entity['nearby_entities'] = self._find_nearby_entities(entity, entities)
            
            enhanced_entities.append(enhanced_entity)
        
        return enhanced_entities

    def _find_nearby_entities(self, target_entity, all_entities):
        """Find entities that appear near the target entity."""
        nearby = []
        target_start = target_entity['start']
        target_end = target_entity['end']
        
        for entity in all_entities:
            if entity == target_entity:
                continue
            
            # Check if entity is within 200 characters
            if (abs(entity['start'] - target_end) <= 200 or 
                abs(target_start - entity['end']) <= 200):
                nearby.append({
                    'text': entity['text'],
                    'type': entity['type'],
                    'distance': min(abs(entity['start'] - target_end), abs(target_start - entity['end']))
                })
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance'])
        return nearby[:3]  # Return top 3 nearest entities


class StreamlitEntityLinker:
    """
    Streamlit wrapper for the ImprovedBERTEntityLinker class.
    
    Provides a web interface with additional visualization and
    export capabilities for entity analysis.
    """
    
    def __init__(self):
        """Initialize the Streamlit Entity Linker."""
        self.entity_linker = ImprovedBERTEntityLinker()
        
        # Initialize session state
        if 'entities' not in st.session_state:
            st.session_state.entities = []
        if 'processed_text' not in st.session_state:
            st.session_state.processed_text = ""
        if 'html_content' not in st.session_state:
            st.session_state.html_content = ""
        if 'analysis_title' not in st.session_state:
            st.session_state.analysis_title = "text_analysis"
        if 'last_processed_hash' not in st.session_state:
            st.session_state.last_processed_hash = ""

    @st.cache_data
    def cached_extract_entities(_self, text: str) -> List[Dict[str, Any]]:
        """Cached entity extraction to avoid reprocessing same text."""
        return _self.entity_linker.extract_entities(text)

    def render_header(self):
        """Render the application header with logo."""
        # Display logo if it exists
        try:
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
        except Exception:
            pass
        
        # Add some spacing after logo
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main title and description
        st.header("From Text to Linked Data using BERT-large-NER")
        st.markdown("**Extract and link named entities using the BERT-large-NER model (no confidence filtering)**")
        
        # Create a simple process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>BERT-large-NER (No Filtering)</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Link to Knowledge Bases:</strong>
                </div>
                <div style="margin: 15px 0;">
                    <div style="background-color: #EFCA89; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Wikidata</strong><br><small>Structured knowledge</small>
                    </div>
                    <div style="background-color: #C3B5AC; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                        <strong>Wikipedia</strong><br><small>Encyclopedia articles</small>
                    </div>
                    <div style="background-color: #BF7B69; padding: 8px; border-radius: 5px; display: inline-block; margin: 3px; font-size: 0.9em;">
                         <strong>Geocoding</strong><br><small>Coordinates & locations</small>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with model information."""
        st.sidebar.subheader("Model Information")
        st.sidebar.info("""
        **NER Model**: BERT-large-NER (dslim/bert-large-NER)
        
        **Configuration**: 
        - No confidence thresholds
        - Simple aggregation strategy
        - All entities included
        
        **Linking**: Wikidata, Wikipedia, OpenStreetMap
        
        **Features**: 
        - All BERT predictions included
        - Enhanced contextual analysis
        - Geographic context detection
        - Entity relationship mapping
        """)
        
        st.sidebar.subheader("Entity Types")
        st.sidebar.info("""
        **BERT NER Types**:
        - PERSON (people)
        - ORG (organizations)  
        - GPE (geopolitical entities)
        - LOC (locations)
        - MISC (miscellaneous)
        
        **Enhancement**:
        - Contextual keywords
        - Geographic coordinates
        - Entity relationships
        """)

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        # Add title input
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Default text
        default_text = """Recording the Whitechapel Pavilion in 1961. 191-193 Whitechapel Road. theatre. It was a dauntingly complex task, as to my (then) untrained eye, it appeared to be an impenetrable forest of heavy timbers, movable platforms and hoisting gear, looking like the combined wreckage of half a dozen windmills! I started by chalking an individual number on every stage joist in an attempt to provide myself with a simple skeleton on which to hang the more complicated details. Richard Southern's explanations enabled me to allocate names to the various pieces of apparatus, correcting my guesses. ('Stage basement' for example was, I learned, an imprecise way of naming a space with three distinct levels). He also gave me a brilliant introduction to the workings of a traditional wood stage and to the theatric purposes each part fulfilled. The attached sketch attempts to give a summary view of the entire substage. It is set at the first level below the stage, with the proscenium wall at the top and the back wall of the stage house at the bottom. In the terminology of the traditional wood stage, this is the 'mezzanine', from which level, all the substage machinery was worked by an army of stage hands. In the centre, the heavily outlined rectangle is the 'cellar', deeper by about 7ft below the mezzanine floor. Housed in the cellar are a variety of vertically movable platforms designed to move pieces of scenery and complete set pieces. It may be observed at this point that not all of this apparatus will have resulted from one build. A wood stage had the great advantage that it could be adapted at short notice by the stage carpenter to meet the demands of a particular production. The substage, as seen, represents a particular moment in its active life. There are five fast rise or 'star' traps for the sudden appearance (or disappearance) of individual performers (clowns, etc) through the stage floor. The three traps nearest to the audience are 'two post' traps, rather primitive and capable of causing serious injury to an inexpert user. Upstage of these are two of the more advanced and marginally safer 'four post' traps. In both types, the performer stood on a box-like counter-weighted platform with his (usually his) head touching the centre of a 'star' of leather-hinged wood segments. Beefy stage hands pulled suddenly (but with split second timing) on the lines supporting the box, shooting him through the star. In an instant, it closed behind him, leaving no visible aperture in the stage surface. Farther upstage is a row of 'sloats', designed to hold scenic flats, to be slid up through the stage floor. Next comes a grave trap which, as the name suggests, can provide a rectangular sinking in the stage ('Alas, poor Yorick'). Finally, a short bridge and a long bridge, to carry heavy set pieces, with or without chorus members, up through (and, when required, a bit above) the stage. These bridges were operated from whopping great drum and shaft mechanisms on the mezzanine. In order to get all these vertical movements to pass through the stage, its joists, counter-intuitively, have to span from side to side, the long span rather than the more obvious short span. This makes it possible to have removable sections ('sliders') in the stage floor, which are held level position by paddle levers at the ends. When these are released, the slider drops on to runners on the sides of the joists and are then winched off to left and right. The survey of the Pavilion stage was important at the time because it seemed to be the first time that anything of the kind had been done, however imperfectly. Since then, we have learned of complete surviving complexes at, for example, Her Majesty's theatre in London, the Citizens in Glasgow and, most importantly, the Tyne theatre in Newcastle, which has been restored to full working order twice (once after a dreadfully destructive fire) by Dr David Wilmore. Nevertheless, the loss of the archaeological evidence of the Pavilion is much to be regretted."""
        
        # Text input area
        text_input = st.text_area(
            "Enter your text here:",
            value=default_text,
            height=300,
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content"
        )
        
        # File upload option
        with st.expander("Or upload a text file"):
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'md'],
                help="Upload a plain text file (.txt) or Markdown file (.md)"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_text = str(uploaded_file.read(), "utf-8")
                    text_input = uploaded_text
                    st.success(f"File uploaded successfully! ({len(uploaded_text)} characters)")
                    # Set default title from filename if no title provided
                    if not analysis_title:
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Use suggested title if available
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title:
            analysis_title = "whitechapel_pavilion_analysis"
        
        return text_input, analysis_title

    def process_text(self, text: str, title: str):
        """Process the input text using the ImprovedBERTEntityLinker."""
        if not text.strip():
            st.warning("Please enter some text to analyse.")
            return
        
        # Check if we've already processed this exact text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text with BERT-large-NER model (no filtering)..."):
            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract entities
                status_text.text("Extracting entities with BERT-large-NER...")
                progress_bar.progress(20)
                entities = self.cached_extract_entities(text)
                
                if not entities:
                    st.warning("No entities found. The BERT model may not have detected any named entities in this text.")
                    return
                
                # Step 2: Enhance with context
                status_text.text("Enhancing entities with contextual information...")
                progress_bar.progress(35)
                entities = self.entity_linker.enhance_entities_with_context(entities, text)
                
                # Step 3: Link to Wikidata
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(60)
                entities = self.entity_linker.link_to_wikidata(entities)
                
                # Step 4: Link to Wikipedia
                status_text.text("Linking to Wikipedia...")
                progress_bar.progress(75)
                entities = self.entity_linker.link_to_wikipedia(entities)
                
                # Step 5: Get coordinates
                status_text.text("Getting coordinates and geocoding...")
                progress_bar.progress(90)
                entities = self.entity_linker.get_coordinates(entities)
                
                # Step 6: Generate visualization
                status_text.text("Generating visualization...")
                progress_bar.progress(100)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store in session state
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Processing complete! Found {len(entities)} entities.")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create HTML content with highlighted entities and enhanced tooltips."""
        import html as html_module
        
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Start with escaped text
        highlighted = html_module.escape(text)
        
        # Replace entities from end to start
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = self.entity_linker.colors.get(entity['type'], '#E7E2D2')
            
            # Create enhanced tooltip
            tooltip_parts = [f"Type: {entity['type']}"]
            
            if entity.get('confidence'):
                tooltip_parts.append(f"Confidence: {float(entity['confidence']):.2f}")
            
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            elif entity.get('wikipedia_description'):
                tooltip_parts.append(f"Description: {entity['wikipedia_description']}")
            
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            if entity.get('nearby_entities'):
                nearby_texts = [e['text'] for e in entity['nearby_entities'][:2]]
                tooltip_parts.append(f"Nearby: {', '.join(nearby_texts)}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Create highlighted span with link
            if entity.get('wikipedia_url'):
                url = html_module.escape(entity["wikipedia_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border-left: 3px solid #2E7D32;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border-left: 3px solid #1976D2;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
                # Just highlight
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; border-left: 3px solid #757575;" title="{tooltip}">{escaped_entity_text}</span>'
            
            # Calculate positions in escaped text
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            # Replace in the escaped text
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    def render_results(self):
        """Render the results section with entities and visualizations."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Show quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entities", len(entities))
        with col2:
            linked_count = len([e for e in entities if any(e.get(key) for key in ['wikidata_url', 'wikipedia_url'])])
            st.metric("Linked Entities", linked_count)
        with col3:
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded Places", geocoded_count)
        with col4:
            person_count = len([e for e in entities if e['type'] == 'PERSON'])
            st.metric("People Found", person_count)
        
        # Highlighted text
        st.subheader("Highlighted Text")
        if st.session_state.html_content:
            st.markdown(
                st.session_state.html_content,
                unsafe_allow_html=True
            )
        else:
            st.info("No highlighted text available. Process some text first.")
        
        # Entity analysis tabs
        tab1, tab2, tab3 = st.tabs(["Entity Summary", "Detailed Analysis", "Export Options"])
        
        with tab1:
            self.render_entity_summary(entities)
        
        with tab2:
            self.render_detailed_analysis(entities)
        
        with tab3:
            self.render_export_section(entities)

    def render_entity_summary(self, entities: List[Dict[str, Any]]):
        """Render a summary table of entities."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Prepare data for table
        summary_data = []
        for entity in entities:
            links = []
            if entity.get('wikipedia_url'):
                links.append("Wikipedia")
            if entity.get('wikidata_url'):
                links.append("Wikidata")
            
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Confidence': f"{float(entity.get('confidence', 0)):.2f}",
                'Links': " | ".join(links) if links else "None",
                'Geocoded': "Yes" if entity.get('latitude') else "No",
                'Context': entity.get('sentence_context', '')[:100] + "..." if entity.get('sentence_context', '') else ""
            }
            summary_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

    def render_detailed_analysis(self, entities: List[Dict[str, Any]]):
        """Render detailed analysis of entities."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + list(set(e['type'] for e in entities))
            )
        
        with col2:
            link_filter = st.selectbox(
                "Filter by Links",
                ["All", "Linked", "Not Linked"]
            )
        
        # Apply filters
        filtered_entities = entities
        if type_filter != "All":
            filtered_entities = [e for e in filtered_entities if e['type'] == type_filter]
        if link_filter == "Linked":
            filtered_entities = [e for e in filtered_entities if any(e.get(key) for key in ['wikidata_url', 'wikipedia_url'])]
        elif link_filter == "Not Linked":
            filtered_entities = [e for e in filtered_entities if not any(e.get(key) for key in ['wikidata_url', 'wikipedia_url'])]
        
        # Display detailed information for each entity
        for i, entity in enumerate(filtered_entities):
            with st.expander(f"{entity['text']} ({entity['type']})", expanded=i < 3):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- Type: {entity['type']}")
                    st.write(f"- Confidence: {float(entity.get('confidence', 0)):.2f}")
                    st.write(f"- Position: {entity['start']}-{entity['end']}")
                    
                    if entity.get('context_keywords'):
                        st.write("**Context Keywords:**")
                        st.write(", ".join(entity['context_keywords']))
                
                with col2:
                    st.write("**External Links:**")
                    if entity.get('wikipedia_url'):
                        st.write(f"- [Wikipedia]({entity['wikipedia_url']})")
                    if entity.get('wikidata_url'):
                        st.write(f"- [Wikidata]({entity['wikidata_url']})")
                    
                    if entity.get('latitude'):
                        st.write("**Location:**")
                        st.write(f"- Coordinates: {entity['latitude']:.4f}, {entity['longitude']:.4f}")
                        st.write(f"- Address: {entity.get('location_name', 'N/A')}")
                
                if entity.get('sentence_context'):
                    st.write("**Sentence Context:**")
                    st.write(f"_{entity['sentence_context']}_")
                
                if entity.get('nearby_entities'):
                    st.write("**Nearby Entities:**")
                    for nearby in entity['nearby_entities']:
                        st.write(f"- {nearby['text']} ({nearby['type']}) - {nearby['distance']} chars away")

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options for the results."""
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_data = {
                "title": st.session_state.analysis_title,
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "model": "dslim/bert-large-NER",
                "settings": "no confidence filtering, simple aggregation",
                "entities": []
            }
            
            # Format entities for JSON
            for entity in entities:
                entity_data = {
                    "text": entity['text'],
                    "type": entity['type'],
                    "start": entity['start'],
                    "end": entity['end'],
                    "confidence": entity.get('confidence', 0),
                    "contextualInformation": {
                        "sentenceContext": entity.get('sentence_context', ''),
                        "contextKeywords": entity.get('context_keywords', []),
                        "nearbyEntities": entity.get('nearby_entities', []),
                        "entityFrequency": entity.get('entity_frequency', 1)
                    }
                }
                
                # Add links
                if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                    entity_data['externalLinks'] = {}
                    if entity.get('wikidata_url'):
                        entity_data['externalLinks']['wikidata'] = entity['wikidata_url']
                    if entity.get('wikipedia_url'):
                        entity_data['externalLinks']['wikipedia'] = entity['wikipedia_url']
                
                # Add geographical information
                if entity.get('latitude') and entity.get('longitude'):
                    entity_data['coordinates'] = {
                        "latitude": entity['latitude'],
                        "longitude": entity['longitude'],
                        "locationName": entity.get('location_name', ''),
                        "source": entity.get('geocoding_source', '')
                    }
                
                json_data['entities'].append(entity_data)
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # HTML export
            if st.session_state.html_content:
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Entity Analysis: {st.session_state.analysis_title}</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #F5F0DC; }}
                        .header {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #dee2e6; }}
                        .content {{ background: white; padding: 25px; border-radius: 10px; line-height: 1.8; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                        .stat-box {{ text-align: center; padding: 15px; background: #e9ecef; border-radius: 8px; }}
                        .legend {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                        .legend-item {{ display: inline-block; margin: 5px 10px; }}
                        .legend-color {{ width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px; }}
                        .note {{ margin: 20px 0; padding: 15px; background: #e8f5e8; border-radius: 8px; border-left: 4px solid #4caf50; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>BERT NER Analysis: {st.session_state.analysis_title}</h1>
                        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <div class="stats">
                            <div class="stat-box">
                                <strong>{len(entities)}</strong><br>Total Entities
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if e.get('latitude')])}</strong><br>Geocoded Places
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if any(e.get(key) for key in ['wikidata_url', 'wikipedia_url'])])}</strong><br>Linked Entities
                            </div>
                        </div>
                    </div>
                    <div class="note">
                        <h3>Model Configuration:</h3>
                        <p>• <strong>Model</strong>: dslim/bert-large-NER</p>
                        <p>• <strong>Aggregation</strong>: Simple strategy</p>
                        <p>• <strong>Filtering</strong>: No confidence thresholds applied</p>
                        <p>• <strong>Linking</strong>: Wikidata, Wikipedia, OpenStreetMap</p>
                    </div>
                    <div class="legend">
                        <h3>Entity Types:</h3>
                        {''.join([f'<div class="legend-item"><span class="legend-color" style="background-color: {self.entity_linker.colors.get(entity_type, "#E7E2D2")};"></span>{entity_type}</div>' for entity_type in sorted(set(e["type"] for e in entities))])}
                    </div>
                    <div class="content">
                        {st.session_state.html_content}
                    </div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="Download HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_entities.html",
                    mime="text/html",
                    use_container_width=True
                )

    def run(self):
        """Main application runner."""
        # Add custom CSS
        st.markdown("""
        <style>
        .stApp {
            background-color: #F5F0DC !important;
        }
        .main .block-container {
            background-color: #F5F0DC !important;
        }
        .stButton > button {
            background-color: #C4A998 !important;
            color: black !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            background-color: #B5998A !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render components
        self.render_header()
        self.render_sidebar()
        
        # Input section
        text_input, analysis_title = self.render_input_section()
        
        # Process button
        if st.button("Process Text", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyse.")
        
        st.markdown("---")
        
        # Results section
        self.render_results()


def main():
    """Main function to run the Streamlit application."""
    try:
        app = StreamlitEntityLinker()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please ensure all required packages are installed:")
        st.code("""
        pip install streamlit streamlit-authenticator
        pip install transformers torch
        pip install pandas plotly
        pip install requests
        pip install PyYAML pycountry
        """)


if __name__ == "__main__":
    main()

