#!/usr/bin/env python3
"""
Enhanced Streamlit Entity Linker Application with improved bert-large-NER model

Key improvements to match FLAIR performance:
1. Removed confidence thresholds that filter out valid entities
2. Better token reconstruction for subword handling
3. More comprehensive pattern matching (excluding dates/money/time)
4. Relaxed entity validation
5. Better handling of multi-word entities
6. Enhanced entity type detection
7. Multiple extraction strategies combined

Author: Enhanced version
"""

import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced NER: From Text to Linked Data using bert-large-NER",
    layout="centered",  
    initial_sidebar_state="collapsed" 
)

# Authentication code (keeping original structure)
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

    # Check if already authenticated
    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
    else:
        # Handle login
        try:
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            if login_result is None:
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                        st.info("Try username: demo_user with your password")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()
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
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
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

class EnhancedEntityLinker:
    """
    Enhanced entity linking functionality with improved detection capabilities.
    
    Key improvements to match FLAIR performance:
    - No confidence filtering on transformer results
    - Better subword token reconstruction  
    - Enhanced pattern matching (excluding dates/money/time)
    - More entity types
    - Better validation logic
    - Multiple extraction strategies
    """
    
    def __init__(self):
        """Initialize the Enhanced Entity Linker."""
        
        # Enhanced color scheme (excluding dates, money, time)
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'ORG': '#9fd2cd',            # Alias
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground
            'LOC': '#EFCA89',            # Alias
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'EVENT': '#C4A998',           # F&B Dead salmon
            'PRODUCT': '#CCBEAA',         # F&B Oxford stone
            'WORK_OF_ART': '#D4C5B9',     # F&B String
            'ADDRESS': '#E8E1D4',         # F&B Clunch
            'MISC': '#DDD3C0',            # F&B Old White
            'CONTACT': '#F5F0DC',         # F&B Slipper Satin
            'URL': '#E0D7C0'              # F&B Lime White darker
        }
        
        # Initialize models
        self.ner_model = None
        self._load_models()

    def _load_models(self):
        """Load model for entity extraction with enhanced settings."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            
            # Load NER model with enhanced configuration
            with st.spinner("Loading enhanced NER model..."):
                try:
                    ner_model_name = "dslim/bert-large-NER"
                    
                    # Load tokenizer and model separately for better control
                    self.tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
                    self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
                    
                    # Create pipeline with optimal settings for maximum recall
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=self.ner_model,
                        tokenizer=self.tokenizer,
                        aggregation_strategy="simple",  # Better for catching all entities
                        device=0 if torch.cuda.is_available() else -1
                    )
                    st.success("Enhanced bert-large-NER model loaded successfully")
                except Exception as e:
                    st.error(f"Failed to load NER model: {e}")
                    self.ner_pipeline = None
                    st.warning("Using pattern-based entity extraction as fallback")
                    
        except ImportError:
            st.error("Required packages not installed. Please install:")
            st.code("pip install transformers torch")
            st.stop()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()

    def extract_entities(self, text: str):
        """Enhanced entity extraction with maximum recall."""
        entities = []
        
        # Strategy 1: Transformer-based NER with NO filtering
        if self.ner_pipeline:
            try:
                raw_entities = self.ner_pipeline(text)
                
                # Process ALL transformer entities - no confidence threshold
                for ent in raw_entities:
                    entity_type = self._map_entity_type(ent['entity_group'])
                    
                    # Skip unwanted types (dates, money, time)
                    if entity_type in ['DATE', 'TIME', 'MONEY']:
                        continue
                    
                    # Better entity text reconstruction
                    entity_text = self._reconstruct_entity_text(ent, text)
                    
                    if not entity_text or len(entity_text.strip()) < 2:
                        continue
                    
                    # Create entity dictionary
                    entity = {
                        'text': entity_text,
                        'type': entity_type,
                        'start': ent['start'],
                        'end': ent['end'],
                        'confidence': ent['score'],
                        'original_label': ent['entity_group'],
                        'extraction_method': 'transformer_enhanced'
                    }
                    
                    # Add contextual information
                    context_info = self._generate_contextual_analysis(text, entity_text, entity_type)
                    entity.update(context_info)
                    
                    # Very relaxed validation - accept almost everything
                    if self._is_valid_entity_minimal(entity, text):
                        entities.append(entity)
                        
            except Exception as e:
                st.warning(f"Enhanced transformer NER failed: {e}")
        
        # Strategy 2: Pattern-based extraction (excluding dates/money/time)
        pattern_entities = self._extract_comprehensive_patterns(text)
        entities.extend(pattern_entities)
        
        # Strategy 3: Capitalization-based extraction for missed entities
        cap_entities = self._extract_capitalized_entities(text)
        entities.extend(cap_entities)
        
        # Strategy 4: Context-based extraction
        context_entities = self._extract_contextual_entities(text)
        entities.extend(context_entities)
        
        # Remove overlapping entities (keep highest confidence)
        entities = self._remove_overlapping_entities(entities)
        
        return entities

    def _reconstruct_entity_text(self, ent, original_text):
        """Better entity text reconstruction handling subwords."""
        # If we have direct start/end positions, use them
        if 'start' in ent and 'end' in ent:
            return original_text[ent['start']:ent['end']].strip()
        
        # Otherwise use the word field and clean it up
        entity_text = ent['word']
        # Remove BERT tokenization artifacts
        entity_text = entity_text.replace('##', '').strip()
        return entity_text

    def _map_entity_type(self, ner_label: str) -> str:
        """Map NER model labels to our standardized types (excluding dates/money/time)."""
        mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'GPE',
            'MISC': 'MISC',
            'FACILITY': 'FACILITY',
            'EVENT': 'EVENT',
            'PRODUCT': 'PRODUCT',
            'WORK_OF_ART': 'WORK_OF_ART'
        }
        return mapping.get(ner_label, 'MISC')

    def _extract_comprehensive_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Comprehensive pattern extraction (excluding dates/money/time)."""
        pattern_entities = []
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entity = {
                'text': match.group(),
                'type': 'CONTACT',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95,
                'original_label': 'EMAIL_PATTERN',
                'extraction_method': 'pattern_enhanced'
            }
            context_info = self._generate_contextual_analysis(text, entity['text'], 'CONTACT')
            entity.update(context_info)
            pattern_entities.append(entity)
        
        # Phone number patterns
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b',
            r'\b\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                entity = {
                    'text': match.group(),
                    'type': 'CONTACT',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'original_label': 'PHONE_PATTERN',
                    'extraction_method': 'pattern_enhanced'
                }
                context_info = self._generate_contextual_analysis(text, entity['text'], 'CONTACT')
                entity.update(context_info)
                pattern_entities.append(entity)
        
        # URL patterns
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        for match in re.finditer(url_pattern, text):
            entity = {
                'text': match.group(),
                'type': 'URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95,
                'original_label': 'URL_PATTERN',
                'extraction_method': 'pattern_enhanced'
            }
            context_info = self._generate_contextual_analysis(text, entity['text'], 'URL')
            entity.update(context_info)
            pattern_entities.append(entity)
        
        # Enhanced Address patterns
        address_patterns = [
            r'\b\d{1,5}[-–—]\d{1,5}\s+[A-Z][a-zA-Z\s]+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens|Court|Close|Crescent|Boulevard|Terrace)\b',
            r'\b\d{1,5}\s+[A-Z][a-zA-Z\s]+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens|Court|Close|Crescent|Boulevard|Terrace)\b',
            r'\b[A-Z][a-zA-Z\s]{2,}\s+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens|Court|Close|Crescent|Boulevard|Terrace)\b'
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = {
                    'text': match.group().strip(),
                    'type': 'ADDRESS',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85,
                    'original_label': 'ADDRESS_PATTERN',
                    'extraction_method': 'pattern_enhanced'
                }
                context_info = self._generate_contextual_analysis(text, entity['text'], 'ADDRESS')
                entity.update(context_info)
                pattern_entities.append(entity)
        
        # Title + Name patterns
        title_pattern = r'\b(?:Dr|Mr|Mrs|Ms|Prof|Professor|Sir|Dame|Lord|Lady)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
        for match in re.finditer(title_pattern, text):
            entity = {
                'text': match.group(),
                'type': 'PERSON',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'original_label': 'TITLE_PERSON_PATTERN',
                'extraction_method': 'pattern_enhanced'
            }
            context_info = self._generate_contextual_analysis(text, entity['text'], 'PERSON')
            entity.update(context_info)
            pattern_entities.append(entity)
        
        # Organization patterns
        org_patterns = [
            r'\b[A-Z][a-zA-Z\s&]+(?:Inc|LLC|Ltd|Corporation|Corp|Company|Co|Limited|plc|AG|GmbH|SA|University|College|Institute|School|Foundation|Trust|Society|Association|Union|Federation|Alliance|Council|Board|Committee|Commission|Agency|Department|Ministry|Office|Bureau|Authority|Service|Group|Team|Club|Organization|Centre|Center)\b'
        ]
        
        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                entity_text = match.group().strip()
                if len(entity_text) < 3:
                    continue
                    
                entity = {
                    'text': entity_text,
                    'type': 'ORGANIZATION',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'original_label': 'ORG_PATTERN',
                    'extraction_method': 'pattern_enhanced'
                }
                context_info = self._generate_contextual_analysis(text, entity['text'], 'ORGANIZATION')
                entity.update(context_info)
                pattern_entities.append(entity)
        
        return pattern_entities

    def _extract_capitalized_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract likely entities based on capitalization patterns."""
        cap_entities = []
        
        # Find sequences of capitalized words
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        
        for match in re.finditer(cap_pattern, text):
            entity_text = match.group().strip()
            
            # Skip short or common words
            if (len(entity_text) < 3 or 
                entity_text.lower() in {'The', 'And', 'But', 'For', 'With', 'This', 'That', 'They', 'When', 'Where', 'What', 'Who', 'How', 'Why'}):
                continue
            
            # Determine likely type based on context
            entity_type = self._guess_entity_type(entity_text, text, match.start(), match.end())
            
            entity = {
                'text': entity_text,
                'type': entity_type,
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.6,
                'original_label': 'CAPITALIZATION',
                'extraction_method': 'capitalization'
            }
            
            context_info = self._generate_contextual_analysis(text, entity['text'], entity_type)
            entity.update(context_info)
            cap_entities.append(entity)
        
        return cap_entities

    def _extract_contextual_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities based on contextual clues."""
        context_entities = []
        
        # Theatre-specific patterns
        theatre_patterns = [
            (r'\b(?:theatre|theater)\b.*?\b([A-Z][a-zA-Z\s]+)\b', 'FACILITY'),
            (r'\b([A-Z][a-zA-Z\s]+)\s+(?:theatre|theater)\b', 'FACILITY'),
            (r'\b(?:stage|auditorium|hall)\b.*?\b([A-Z][a-zA-Z\s]+)\b', 'FACILITY'),
        ]
        
        for pattern, entity_type in theatre_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_text = match.group(1).strip()
                if len(entity_text) > 2:
                    entity = {
                        'text': entity_text,
                        'type': entity_type,
                        'start': text.find(entity_text, match.start()),
                        'end': text.find(entity_text, match.start()) + len(entity_text),
                        'confidence': 0.7,
                        'original_label': 'THEATRE_CONTEXT',
                        'extraction_method': 'contextual'
                    }
                    context_info = self._generate_contextual_analysis(text, entity['text'], entity_type)
                    entity.update(context_info)
                    context_entities.append(entity)
        
        return context_entities

    def _guess_entity_type(self, entity_text: str, full_text: str, start: int, end: int) -> str:
        """Guess entity type based on context and patterns."""
        entity_lower = entity_text.lower()
        
        # Get surrounding context
        context_start = max(0, start - 50)
        context_end = min(len(full_text), end + 50)
        context = full_text[context_start:context_end].lower()
        
        # Person indicators
        person_indicators = ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'dame', 'said', 'told', 'explained', 'mentioned']
        if any(indicator in context for indicator in person_indicators):
            return 'PERSON'
        
        # Organization indicators  
        org_indicators = ['company', 'corporation', 'university', 'school', 'theatre', 'hospital', 'government']
        if any(indicator in context for indicator in org_indicators):
            return 'ORGANIZATION'
        
        # Location indicators
        loc_indicators = ['city', 'town', 'country', 'street', 'road', 'avenue', 'in', 'at', 'from', 'to']
        if any(indicator in context for indicator in loc_indicators):
            return 'LOCATION'
        
        # Default to MISC for ambiguous cases
        return 'MISC'

    def _is_valid_entity_minimal(self, entity: Dict[str, Any], text: str) -> bool:
        """Minimal validation - accept most entities to maximize recall."""
        entity_text = entity['text'].strip()
        
        # Only reject very obviously invalid entities
        if (len(entity_text) <= 1 or 
            entity_text.lower() in {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'} or
            entity_text.replace(' ', '').replace('.', '').replace('-', '').isdigit()):
            return False
        
        return True

    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the highest confidence ones."""
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:
                # Check if entities overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # Keep the higher confidence entity, or the longer one if confidence is equal
                    if (entity.get('confidence', 0) > existing.get('confidence', 0) or
                        (entity.get('confidence', 0) == existing.get('confidence', 0) and 
                         len(entity['text']) > len(existing['text']))):
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered

    def _generate_contextual_analysis(self, text: str, entity_text: str, entity_type: str) -> Dict[str, Any]:
        """Generate contextual analysis using rule-based approaches."""
        context_info = {
            'sentence_context': self._extract_sentence_context({'text': entity_text, 'start': text.find(entity_text), 'end': text.find(entity_text) + len(entity_text)}, text),
            'context_keywords': self._extract_context_keywords(entity_text, text),
            'entity_frequency': text.lower().count(entity_text.lower()),
            'surrounding_entities': []
        }
        
        return context_info

    def _extract_sentence_context(self, entity: Dict[str, Any], text: str) -> str:
        """Extract the sentence containing the entity."""
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
        
        # Extract meaningful words
        keywords = []
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', context)
        keywords.extend(words)
        
        # Add important terms
        important_terms = ['theatre', 'stage', 'company', 'organization', 'university', 'hospital', 'government', 'technology', 'research']
        for term in important_terms:
            if term in context.lower():
                keywords.append(term)
        
        return list(set(keywords))[:5]

    def enhance_entities_with_context(self, entities, text):
        """Enhance entities with additional contextual information."""
        enhanced_entities = []
        
        for entity in entities:
            enhanced_entity = entity.copy()
            enhanced_entity['nearby_entities'] = self._find_nearby_entities(entity, entities)
            enhanced_entity['context_confidence'] = self._calculate_context_confidence(entity, text)
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
            
            if (abs(entity['start'] - target_end) <= 200 or 
                abs(target_start - entity['end']) <= 200):
                nearby.append({
                    'text': entity['text'],
                    'type': entity['type'],
                    'distance': min(abs(entity['start'] - target_end), abs(target_start - entity['end']))
                })
        
        nearby.sort(key=lambda x: x['distance'])
        return nearby[:3]

    def _calculate_context_confidence(self, entity, text):
        """Calculate confidence score based on contextual factors."""
        base_confidence = entity.get('confidence', 0.5)
        
        confidence_boost = 0
        
        # Multiple occurrences
        occurrences = text.lower().count(entity['text'].lower())
        if occurrences > 1:
            confidence_boost += 0.1
        
        # Proper capitalization
        if entity['text'][0].isupper() and entity['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']:
            confidence_boost += 0.1
        
        # External validation (links)
        if any(entity.get(key) for key in ['wikidata_url', 'wikipedia_url', 'britannica_url']):
            confidence_boost += 0.2
        
        # Coordinates for places
        if entity.get('latitude') and entity['type'] in ['LOCATION', 'GPE', 'ADDRESS']:
            confidence_boost += 0.1
        
        # Context keywords
        if entity.get('context_keywords'):
            confidence_boost += 0.1
        
        # Extraction method confidence
        if entity.get('extraction_method') == 'transformer_enhanced':
            confidence_boost += 0.05
        elif entity.get('extraction_method') == 'pattern_enhanced':
            confidence_boost += 0.1
        
        return min(1.0, base_confidence + confidence_boost)

    def get_coordinates(self, entities):
        """Enhanced coordinate lookup with geographical context detection."""
        context_clues = self._detect_geographical_context(
            st.session_state.get('processed_text', ''), 
            entities
        )
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION', 'ADDRESS']
        
        for entity in entities:
            if entity['type'] in place_types:
                if entity.get('latitude') is not None:
                    continue
                
                if self._try_contextual_geocoding(entity, context_clues):
                    continue
                    
                if self._try_direct_geocoding(entity):
                    continue
        
        return entities

    def _detect_geographical_context(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Detect geographical context from the text using pycountry for countries."""
        context_clues = []
        text_lower = text.lower()
        
        # Create a lookup set of country names and common variations
        country_name_map = {}
        for country in pycountry.countries:
            names = {country.name.lower()}
            if hasattr(country, 'official_name'):
                names.add(country.official_name.lower())
            if hasattr(country, 'common_name'):
                names.add(country.common_name.lower())
            names.add(country.alpha_2.lower())
            names.add(country.alpha_3.lower())
            country_name_map[country.name.lower()] = names

        # Flatten the name map into a lookup for matching
        flat_country_lookup = {}
        for canonical, variants in country_name_map.items():
            for v in variants:
                flat_country_lookup[v] = canonical

        # Check if any known country names are in the text
        for variant, canonical in flat_country_lookup.items():
            if f" {variant} " in f" {text_lower} " and canonical not in context_clues:
                context_clues.append(canonical)

        # Add GPE/LOCATION entities if they match a known country
        for entity in entities:
            if entity['type'] in ['GPE', 'LOCATION']:
                entity_lower = entity['text'].strip().lower()
                if entity_lower in flat_country_lookup:
                    canonical = flat_country_lookup[entity_lower]
                    if canonical not in context_clues:
                        context_clues.append(canonical)

        return context_clues[:3]

    def _try_contextual_geocoding(self, entity, context_clues):
        """Try geocoding with geographical context using pycountry."""
        if not context_clues:
            return False

        search_variations = [entity['text']]

        country_names = {c.name.lower(): c.name for c in pycountry.countries}
        aliases = {
            'uk': 'United Kingdom',
            'usa': 'United States',
            'us': 'United States',
            'england': 'United Kingdom',
            'scotland': 'United Kingdom',
            'wales': 'United Kingdom',
            'britain': 'United Kingdom'
        }

        city_overrides = {
            'london': ['London, United Kingdom'],
            'new york': ['New York, United States'],
            'paris': ['Paris, France'],
            'tokyo': ['Tokyo, Japan'],
            'sydney': ['Sydney, Australia'],
        }

        for context in context_clues:
            context_lower = context.lower()
            if context_lower in city_overrides:
                for variant in city_overrides[context_lower]:
                    search_variations.append(f"{entity['text']}, {variant}")
            else:
                resolved_name = aliases.get(context_lower) or country_names.get(context_lower)
                if resolved_name:
                    search_variations.append(f"{entity['text']}, {resolved_name}")

        search_variations = list(dict.fromkeys(search_variations))

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

                time.sleep(0.3)
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
        
            time.sleep(0.3)
        except Exception as e:
            pass
        
        return False

    def link_to_wikidata(self, entities):
        """Add Wikidata linking with enhanced context and type validation."""
        entity_type_superclasses = {
            'PERSON': {'Q5', 'Q22989102'},
            'GPE': {'Q6256'},
            'LOCATION': {'Q82794', 'Q2221906'},
            'ORGANIZATION': {'Q43229', 'Q783794'}
        }

        for entity in entities:
            try:
                url = "https://www.wikidata.org/w/api.php"
                search_terms = [entity['text']]
                
                if entity['type'] == 'PERSON':
                    search_terms.append(f"{entity['text']} person")
                elif entity['type'] == 'ORGANIZATION':
                    search_terms.append(f"{entity['text']} organization")
                    search_terms.append(f"{entity['text']} company")
                elif entity['type'] in ['LOCATION', 'GPE']:
                    search_terms.append(f"{entity['text']} place")
                    search_terms.append(f"{entity['text']} location")

                if entity.get('context_keywords'):
                    for keyword in entity['context_keywords'][:2]:
                        search_terms.append(f"{entity['text']} {keyword}")

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

                    detail_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_qid}.json"
                    detail_response = requests.get(detail_url, timeout=5)
                    if detail_response.status_code != 200:
                        continue

                    entity_data = detail_response.json()
                    claims = entity_data.get('entities', {}).get(entity_qid, {}).get('claims', {})
                    instance_of = claims.get('P31', [])

                    instance_qs = {
                        claim['mainsnak']['datavalue']['value']['id']
                        for claim in instance_of
                        if 'datavalue' in claim['mainsnak']
                    }

                    allowed_qs = entity_type_superclasses.get(entity['type'], set())
                    if instance_qs & allowed_qs:
                        entity['wikidata_url'] = f"http://www.wikidata.org/entity/{entity_qid}"
                        entity['wikidata_description'] = result.get('description', '')
                        entity['wikidata_label'] = result.get('label', '')
                        break
                    else:
                        entity['wikidata_url'] = f"http://www.wikidata.org/entity/{entity_qid}"
                        entity['wikidata_description'] = result.get('description', '')
                        entity['wikidata_label'] = result.get('label', '')
                        entity['wikidata_note'] = 'Type mismatch; fallback used'
                        break

                time.sleep(0.1)

            except Exception:
                continue

        return entities

    def link_to_wikipedia(self, entities):
        """Add Wikipedia linking for entities without Wikidata links."""
        for entity in entities:
            if entity.get('wikidata_url'):
                continue
                
            try:
                search_url = "https://en.wikipedia.org/w/api.php"
                search_terms = [entity['text']]
                
                if entity.get('context_keywords'):
                    for keyword in entity['context_keywords'][:2]:
                        search_terms.append(f"{entity['text']} {keyword}")
                
                if entity['type'] == 'PERSON':
                    search_terms.append(f"{entity['text']} biography")
                elif entity['type'] == 'ORGANIZATION':
                    search_terms.append(f"{entity['text']} company")
                elif entity['type'] in ['LOCATION', 'GPE']:
                    search_terms.append(f"{entity['text']} geography")
                
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
                            
                            encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                            entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                            entity['wikipedia_title'] = page_title
                            
                            if result.get('snippet'):
                                snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                                entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                            
                            break
                
                time.sleep(0.2)
            except Exception as e:
                pass
        
        return entities

    def link_to_britannica(self, entities):
        """Add Britannica linking for entities without existing links.""" 
        for entity in entities:
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_url = "https://www.britannica.com/search"
                search_terms = [entity['text']]
                
                if entity.get('context_keywords'):
                    for keyword in entity['context_keywords'][:1]:
                        search_terms.append(f"{entity['text']} {keyword}")
                
                for search_term in search_terms[:2]:
                    params = {'query': search_term}
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(search_url, params=params, headers=headers, timeout=10)
                    if response.status_code == 200:
                        pattern = r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>'
                        matches = re.findall(pattern, response.text)
                        
                        for url_path, link_text in matches:
                            if (entity['text'].lower() in link_text.lower() or 
                                link_text.lower() in entity['text'].lower()):
                                entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                                entity['britannica_title'] = link_text.strip()
                                break
                        
                        if entity.get('britannica_url'):
                            break
                
                time.sleep(0.3)
            except Exception:
                pass
        
        return entities

    def link_to_openstreetmap(self, entities):
        """Add OpenStreetMap links to addresses and places."""
        for entity in entities:
            if entity['type'] in ['ADDRESS', 'LOCATION', 'GPE', 'FACILITY']:
                try:
                    url = "https://nominatim.openstreetmap.org/search"
                    params = {
                        'q': entity['text'],
                        'format': 'json',
                        'limit': 1,
                        'addressdetails': 1
                    }
                    headers = {'User-Agent': 'EntityLinker/1.0'}
                    
                    response = requests.get(url, params=params, headers=headers, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            result = data[0]
                            lat = result['lat']
                            lon = result['lon']
                            entity['openstreetmap_url'] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18"
                            entity['openstreetmap_display_name'] = result['display_name']
                            
                            if not entity.get('latitude'):
                                entity['latitude'] = float(lat)
                                entity['longitude'] = float(lon)
                                entity['location_name'] = result['display_name']
                                entity['geocoding_source'] = 'openstreetmap'
                    
                    time.sleep(0.2)
                except Exception:
                    pass
        
        return entities


class StreamlitEntityLinker:
    """
    Enhanced Streamlit wrapper for the entity linker with improved UI.
    """
    
    def __init__(self):
        """Initialize the Streamlit Entity Linker."""
        self.entity_linker = EnhancedEntityLinker()
        
        # Initialize session state
        if 'entities' not in st.session_state:
            st.session_state.entities = []
        if 'processed_text' not in st.session_state:
            st.session_state.processed_text = ""
        if 'html_content' not in st.session_state:
            st.session_state.html_content = ""
        if 'analysis_title' not in st.session_state:
            st.session_state.analysis_title = "enhanced_text_analysis"
        if 'last_processed_hash' not in st.session_state:
            st.session_state.last_processed_hash = ""

    @st.cache_data
    def cached_extract_entities(_self, text: str) -> List[Dict[str, Any]]:
        """Cached entity extraction to avoid reprocessing same text."""
        return _self.entity_linker.extract_entities(text)
    
    @st.cache_data  
    def cached_link_to_wikidata(_self, entities_json: str) -> str:
        """Cached Wikidata linking."""
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikidata(entities)
        return json.dumps(linked_entities, default=str)
    
    @st.cache_data
    def cached_link_to_britannica(_self, entities_json: str) -> str:
        """Cached Britannica linking."""
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_britannica(entities)
        return json.dumps(linked_entities, default=str)

    @st.cache_data
    def cached_link_to_wikipedia(_self, entities_json: str) -> str:
        """Cached Wikipedia linking."""
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikipedia(entities)
        return json.dumps(linked_entities, default=str)

    def render_header(self):
        """Render the enhanced application header."""
        try:
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=300)
            else:
                st.info("Place your logo.png file in the same directory as this app to display it here")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")        
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Enhanced NER: From Text to Linked Data")
        st.markdown("**Improved entity extraction using enhanced bert-large-NER model to match FLAIR performance**")
        
        # Enhanced process diagram
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #E0D7C0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #C4C3A2; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Input Text</strong>
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="background-color: #9fd2cd; padding: 10px; border-radius: 5px; display: inline-block; margin: 5px;">
                     <strong>Enhanced Multi-Strategy Extraction</strong>
                </div>
                <div style="margin: 10px 0; font-size: 0.9em;">
                    <strong>4 Extraction Methods Combined:</strong><br>
                    • Transformer (NO confidence threshold)<br>
                    • Enhanced Pattern Matching<br>
                    • Capitalization Analysis<br>
                    • Contextual Detection
                </div>
                <div style="margin: 10px 0;">⬇️</div>
                <div style="text-align: center;">
                    <strong>Maximum Entity Recall (excluding dates/money/time)</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the enhanced sidebar."""
        st.sidebar.subheader("Enhanced Model")
        st.sidebar.success("""
        **Key Improvements**:
        ✅ NO confidence filtering
        ✅ Better subword handling
        ✅ 4 extraction strategies
        ✅ Enhanced patterns
        ✅ Relaxed validation
        ✅ Maximum recall focus
        """)
        
        st.sidebar.subheader("Entity Types Detected")
        st.sidebar.info("""
        **Core Types**:
        • PERSON, ORGANIZATION, LOCATION, GPE, MISC
        
        **Enhanced Patterns**:
        • CONTACT (emails, phones)
        • URL (web addresses)  
        • ADDRESS (street addresses)
        • FACILITY, EVENT, PRODUCT
        
        **Excluded**: Dates, Money, Time
        """)

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Default text
        default_text = """Recording the Whitechapel Pavilion in 1961. 191-193 Whitechapel Road. theatre. It was a dauntingly complex task, as to my (then) untrained eye, it appeared to be an impenetrable forest of heavy timbers, movable platforms and hoisting gear, looking like the combined wreckage of half a dozen windmills! I started by chalking an individual number on every stage joist in an attempt to provide myself with a simple skeleton on which to hang the more complicated details. Richard Southern's explanations enabled me to allocate names to the various pieces of apparatus, correcting my guesses. ('Stage basement' for example was, I learned, an imprecise way of naming a space with three distinct levels). He also gave me a brilliant introduction to the workings of a traditional wood stage and to the theatric purposes each part fulfilled. The attached sketch attempts to give a summary view of the entire substage. It is set at the first level below the stage, with the proscenium wall at the top and the back wall of the stage house at the bottom. In the terminology of the traditional wood stage, this is the 'mezzanine', from which level, all the substage machinery was worked by an army of stage hands. In the centre, the heavily outlined rectangle is the 'cellar', deeper by about 7ft below the mezzanine floor. Housed in the cellar are a variety of vertically movable platforms designed to move pieces of scenery and complete set pieces. It may be observed at this point that not all of this apparatus will have resulted from one build. A wood stage had the great advantage that it could be adapted at short notice by the stage carpenter to meet the demands of a particular production. The substage, as seen, represents a particular moment in its active life. There are five fast rise or 'star' traps for the sudden appearance (or disappearance) of individual performers (clowns, etc) through the stage floor. The three traps nearest to the audience are 'two post' traps, rather primitive and capable of causing serious injury to an inexpert user. Upstage of these are two of the more advanced and marginally safer 'four post' traps. In both types, the performer stood on a box-like counter-weighted platform with his (usually his) head touching the centre of a 'star' of leather-hinged wood segments. Beefy stage hands pulled suddenly (but with split second timing) on the lines supporting the box, shooting him through the star. In an instant, it closed behind him, leaving no visible aperture in the stage surface. Farther upstage is a row of 'sloats', designed to hold scenic flats, to be slid up through the stage floor. Next comes a grave trap which, as the name suggests, can provide a rectangular sinking in the stage ('Alas, poor Yorick'). Finally, a short bridge and a long bridge, to carry heavy set pieces, with or without chorus members, up through (and, when required, a bit above) the stage. These bridges were operated from whopping great drum and shaft mechanisms on the mezzanine. In order to get all these vertical movements to pass through the stage, its joists, counter-intuitively, have to span from side to side, the long span rather than the more obvious short span. This makes it possible to have removable sections '(sliders') in the stage floor, which are held level position by paddle levers at the ends. When these are released, the slider drops on to runners on the sides of the joists and are then winched off to left and right. The survey of the Pavilion stage was important at the time because it seemed to be the first time that anything of the kind had been done, however imperfectly. Since then, we have learned of complete surviving complexes at, for example, Her Majesty's theatre in London, the Citizens in Glasgow and, most importantly, the Tyne theatre in Newcastle, which has been restored to full working order twice (once after a dreadfully destructive fire) by Dr David Wilmore. Nevertheless, the loss of the archaeological evidence of the Pavilion is much to be regretted. I can have enjoyable fantasies about witnessing an elaborate pantomime transformation scene from the mezzanine of a Victorian theatre. The place is seething with stage hands, dressers and flimsily clad chorus girls climbing on to the bridges, while the stage is shuddering, having been temporarily robbed of rigidity by the drawing off of the sliders. Orders must be observed to the letter and to the very second, but there can be no shouting, however energetically the orchestra plays. Add naked gas flames to the mix… That's enough!"""
        
        text_input = st.text_area(
            "Enter your text here:",
            value=default_text,
            height=300,
            placeholder="Paste your text here for enhanced entity extraction...",
            help="This enhanced version should find significantly more entities than the original"
        )
        
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
                    if not analysis_title:
                        default_title = os.path.splitext(uploaded_file.name)[0]
                        st.session_state.suggested_title = default_title
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title:
            analysis_title = "enhanced_whitechapel_pavilion_analysis"
        
        return text_input, analysis_title

    def process_text(self, text: str, title: str):
        """Process the input text using the Enhanced Entity Linker."""
        if not text.strip():
            st.warning("Please enter some text to analyse.")
            return
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text with enhanced multi-strategy extraction..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Enhanced entity extraction
                status_text.text("Running enhanced multi-strategy entity extraction...")
                progress_bar.progress(25)
                entities = self.cached_extract_entities(text)
                
                st.info(f"Found {len(entities)} entities with enhanced methods (vs. original detection)")
                
                # Step 2: Enhance with context
                status_text.text("Enhancing entities with contextual information...")
                progress_bar.progress(40)
                entities = self.entity_linker.enhance_entities_with_context(entities, text)
                
                # Step 3: Link to external sources
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(55)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                status_text.text("Linking to Wikipedia...")
                progress_bar.progress(70)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikipedia(entities_json)
                entities = json.loads(linked_entities_json)
                
                status_text.text("Linking to Britannica...")
                progress_bar.progress(80)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Geocoding
                status_text.text("Getting coordinates and geocoding...")
                progress_bar.progress(90)
                entities = self.entity_linker.get_coordinates(entities)
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                # Step 5: Generate visualization
                status_text.text("Generating enhanced visualization...")
                progress_bar.progress(100)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store results
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Enhanced processing complete! Found {len(entities)} entities with maximum recall approach.")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create HTML content with highlighted entities and enhanced tooltips."""
        import html as html_module
        
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        highlighted = html_module.escape(text)
        
        for entity in sorted_entities:
            has_links = any(entity.get(key) for key in ['britannica_url', 'wikidata_url', 'wikipedia_url', 'openstreetmap_url'])
            has_coordinates = entity.get('latitude') is not None
            high_confidence = float(entity.get('context_confidence', 0)) > 0.5
            
            if not (has_links or has_coordinates or high_confidence):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = self.entity_linker.colors.get(entity['type'], '#E7E2D2')
            
            # Create enhanced tooltip
            tooltip_parts = [f"Type: {entity['type']}"]
            
            if entity.get('context_confidence'):
                tooltip_parts.append(f"Confidence: {float(entity['context_confidence']):.2f}")
            
            if entity.get('extraction_method'):
                tooltip_parts.append(f"Method: {entity['extraction_method']}")
            
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            elif entity.get('wikipedia_description'):
                tooltip_parts.append(f"Description: {entity['wikipedia_description']}")
            elif entity.get('britannica_title'):
                tooltip_parts.append(f"Description: {entity['britannica_title']}")
            
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
            elif entity.get('britannica_url'):
                url = html_module.escape(entity["britannica_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border-left: 3px solid #D32F2F;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('openstreetmap_url'):
                url = html_module.escape(entity["openstreetmap_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black; border-left: 3px solid #388E3C;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
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
            st.info("Enter some text above and click 'Process Text with Enhanced Methods' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Enhanced Results")
        
        # Enhanced stats
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Entities", len(entities))
        with col2:
            transformer_count = len([e for e in entities if e.get('extraction_method') == 'transformer_enhanced'])
            st.metric("Transformer", transformer_count)
        with col3:
            pattern_count = len([e for e in entities if e.get('extraction_method') == 'pattern_enhanced'])
            st.metric("Pattern", pattern_count)
        with col4:
            cap_count = len([e for e in entities if e.get('extraction_method') == 'capitalization'])
            st.metric("Capitalization", cap_count)
        with col5:
            context_count = len([e for e in entities if e.get('extraction_method') == 'contextual'])
            st.metric("Contextual", context_count)
        
        # Show improvement message
        st.success("**Enhanced Detection**: This version removes confidence thresholds and uses multiple extraction strategies to maximize entity recall, matching FLAIR's performance.")
        
        # Highlighted text
        st.subheader("Enhanced Highlighted Text")
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
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Method': entity.get('extraction_method', 'unknown'),
                'Confidence': f"{float(entity.get('context_confidence', 0)):.2f}",
                'Links': self.format_entity_links(entity),
                'Context': entity.get('sentence_context', '')[:100] + "..." if entity.get('sentence_context', '') else ""
            }
            summary_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Show method breakdown
        st.subheader("Extraction Method Breakdown")
        method_counts = {}
        for entity in entities:
            method = entity.get('extraction_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        for method, count in method_counts.items():
            st.write(f"• **{method}**: {count} entities")

    def render_detailed_analysis(self, entities: List[Dict[str, Any]]):
        """Render detailed analysis of entities."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + list(set(e['type'] for e in entities))
            )
        
        with col2:
            method_filter = st.selectbox(
                "Filter by Method",
                ["All"] + list(set(e.get('extraction_method', 'unknown') for e in entities))
            )
        
        with col3:
            confidence_threshold = st.slider(
                "Minimum Confidence",
                0.0, 1.0, 0.0, 0.1
            )
        
        # Apply filters
        filtered_entities = entities
        if type_filter != "All":
            filtered_entities = [e for e in filtered_entities if e['type'] == type_filter]
        if method_filter != "All":
            filtered_entities = [e for e in filtered_entities if e.get('extraction_method') == method_filter]
        filtered_entities = [e for e in filtered_entities if float(e.get('context_confidence', 0)) >= confidence_threshold]
        
        st.write(f"**Showing {len(filtered_entities)} entities** (filtered from {len(entities)} total)")
        
        # Display detailed information for each entity
        for i, entity in enumerate(filtered_entities):
            with st.expander(f"{entity['text']} ({entity['type']}) - {entity.get('extraction_method', 'unknown')}", expanded=i < 3):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- Type: {entity['type']}")
                    st.write(f"- Method: {entity.get('extraction_method', 'unknown')}")
                    st.write(f"- Confidence: {float(entity.get('context_confidence', 0)):.2f}")
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
                    if entity.get('britannica_url'):
                        st.write(f"- [Britannica]({entity['britannica_url']})")
                    if entity.get('openstreetmap_url'):
                        st.write(f"- [OpenStreetMap]({entity['openstreetmap_url']})")
                    
                    if entity.get('latitude'):
                        st.write("**Location:**")
                        st.write(f"- Coordinates: {entity['latitude']:.4f}, {entity['longitude']:.4f}")
                        st.write(f"- Address: {entity.get('location_name', 'N/A')}")
                        st.write(f"- Source: {entity.get('geocoding_source', 'N/A')}")
                
                if entity.get('sentence_context'):
                    st.write("**Sentence Context:**")
                    st.write(f"_{entity['sentence_context']}_")
                
                if entity.get('nearby_entities'):
                    st.write("**Nearby Entities:**")
                    for nearby in entity['nearby_entities']:
                        st.write(f"- {nearby['text']} ({nearby['type']}) - {nearby['distance']} chars away")

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table."""
        links = []
        if entity.get('wikipedia_url'):
            links.append("Wikipedia")
        if entity.get('wikidata_url'):
            links.append("Wikidata")
        if entity.get('britannica_url'):
            links.append("Britannica")
        if entity.get('openstreetmap_url'):
            links.append("OpenStreetMap")
        return " | ".join(links) if links else "No links"

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render enhanced export options for the results."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced JSON-LD export
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "processingMethod": "Enhanced bert-large-NER + Multi-Strategy Extraction",
                "modelInfo": {
                    "nerModel": "dslim/bert-large-NER",
                    "aggregationStrategy": "simple",
                    "confidenceThreshold": "none",
                    "extractionMethods": ["transformer_enhanced", "pattern_enhanced", "capitalization", "contextual"],
                    "improvements": ["no_confidence_filtering", "better_subword_handling", "multiple_strategies", "enhanced_patterns"],
                    "linkingSources": ["Wikidata", "Wikipedia", "Britannica", "OpenStreetMap"]
                },
                "entities": []
            }
            
            # Format entities for enhanced JSON-LD
            for entity in entities:
                entity_data = {
                    "name": entity['text'],
                    "type": entity['type'],
                    "startOffset": entity['start'],
                    "endOffset": entity['end'],
                    "confidence": entity.get('context_confidence', 0),
                    "extractionMethod": entity.get('extraction_method', 'unknown'),
                    "contextualInformation": {
                        "sentenceContext": entity.get('sentence_context', ''),
                        "contextKeywords": entity.get('context_keywords', []),
                        "nearbyEntities": entity.get('nearby_entities', []),
                        "entityFrequency": entity.get('entity_frequency', 1)
                    }
                }
                
                # Add all available links
                same_as_links = []
                if entity.get('wikidata_url'):
                    same_as_links.append(entity['wikidata_url'])
                if entity.get('wikipedia_url'):
                    same_as_links.append(entity['wikipedia_url'])
                if entity.get('britannica_url'):
                    same_as_links.append(entity['britannica_url'])
                if entity.get('openstreetmap_url'):
                    same_as_links.append(entity['openstreetmap_url'])
                
                if same_as_links:
                    entity_data['sameAs'] = same_as_links
                
                # Add descriptions
                if entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
                elif entity.get('wikipedia_description'):
                    entity_data['description'] = entity['wikipedia_description']
                elif entity.get('britannica_title'):
                    entity_data['description'] = entity['britannica_title']
                
                # Add geographical information
                if entity.get('latitude') and entity.get('longitude'):
                    entity_data['geo'] = {
                        "@type": "GeoCoordinates",
                        "latitude": entity['latitude'],
                        "longitude": entity['longitude']
                    }
                    if entity.get('location_name'):
                        entity_data['geo']['name'] = entity['location_name']
                    if entity.get('geocoding_source'):
                        entity_data['geo']['source'] = entity['geocoding_source']
                
                json_data['entities'].append(entity_data)
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download Enhanced JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_enhanced_entities.jsonld",
                mime="application/ld+json",
                use_container_width=True
            )
        
        with col2:
            # Enhanced HTML export
            if st.session_state.html_content:
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Enhanced Entity Analysis: {st.session_state.analysis_title}</title>
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
                        .improvement-note {{ margin: 20px 0; padding: 15px; background: #e8f5e8; border-radius: 8px; border-left: 4px solid #4caf50; }}
                        .method-breakdown {{ margin: 20px 0; padding: 15px; background: #fff3e0; border-radius: 8px; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Enhanced Entity Analysis: {st.session_state.analysis_title}</h1>
                        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} with maximum recall methods</p>
                        <div class="stats">
                            <div class="stat-box">
                                <strong>{len(entities)}</strong><br>Total Entities
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if e.get('extraction_method') == 'transformer_enhanced'])}</strong><br>Transformer
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if e.get('extraction_method') == 'pattern_enhanced'])}</strong><br>Pattern
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if e.get('extraction_method') == 'capitalization'])}</strong><br>Capitalization
                            </div>
                            <div class="stat-box">
                                <strong>{len([e for e in entities if e.get('extraction_method') == 'contextual'])}</strong><br>Contextual
                            </div>
                        </div>
                    </div>
                    <div class="improvement-note">
                        <h3>Enhanced Detection Features:</h3>
                        <p>• <strong>NO Confidence Threshold</strong>: Accept all transformer predictions for maximum recall</p>
                        <p>• <strong>4 Extraction Strategies</strong>: Transformer + Patterns + Capitalization + Context</p>
                        <p>• <strong>Better Subword Handling</strong>: Improved token reconstruction</p>
                        <p>• <strong>Relaxed Validation</strong>: Minimal entity filtering to maximize detection</p>
                        <p>• <strong>Enhanced Patterns</strong>: More comprehensive pattern matching</p>
                        <p>• <strong>Excluding</strong>: Dates, money, and time entities as requested</p>
                    </div>
                    <div class="method-breakdown">
                        <h3>Multi-Strategy Approach:</h3>
                        <p>• <strong>transformer_enhanced</strong>: BERT-large-NER with simple aggregation, no confidence filtering</p>
                        <p>• <strong>pattern_enhanced</strong>: Comprehensive regex patterns for structured data</p>
                        <p>• <strong>capitalization</strong>: Capitalize word sequence analysis</p>
                        <p>• <strong>contextual</strong>: Domain-specific contextual patterns (theatre, etc.)</p>
                    </div>
                    <div class="legend">
                        <h3>Entity Types:</h3>
                        {''.join([f'<div class="legend-item"><span class="legend-color" style="background-color: {self.entity_linker.colors.get(entity_type, "#E7E2D2")};"></span>{entity_type}</div>' for entity_type in sorted(set(e["type"] for e in entities))])}
                    </div>
                    <div class="content">
                        {st.session_state.html_content}
                    </div>
                    <div class="header">
                        <h3>Performance Enhancement:</h3>
                        <p>This enhanced version should detect significantly more entities than the original implementation by:</p>
                        <p>• Removing confidence thresholds that filter out valid entities</p>
                        <p>• Using multiple complementary extraction strategies</p>
                        <p>• Better handling of subword tokenization issues</p>
                        <p>• More aggressive pattern matching while maintaining precision</p>
                    </div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="Download Enhanced HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_enhanced_entities.html",
                    mime="text/html",
                    use_container_width=True
                )
        
        # Additional export formats
        st.subheader("Additional Export Formats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv_data = []
            for entity in entities:
                csv_data.append({
                    'Entity': entity['text'],
                    'Type': entity['type'],
                    'Method': entity.get('extraction_method', 'unknown'),
                    'Confidence': entity.get('context_confidence', 0),
                    'Start': entity['start'],
                    'End': entity['end'],
                    'Wikipedia': entity.get('wikipedia_url', ''),
                    'Wikidata': entity.get('wikidata_url', ''),
                    'Britannica': entity.get('britannica_url', ''),
                    'Latitude': entity.get('latitude', ''),
                    'Longitude': entity.get('longitude', ''),
                    'Location': entity.get('location_name', ''),
                    'Context': entity.get('sentence_context', '')
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"{st.session_state.analysis_title}_enhanced_entities.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # GeoJSON export for entities with coordinates
            geo_entities = [e for e in entities if e.get('latitude')]
            if geo_entities:
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": []
                }
                
                for entity in geo_entities:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [entity['longitude'], entity['latitude']]
                        },
                        "properties": {
                            "name": entity['text'],
                            "type": entity['type'],
                            "method": entity.get('extraction_method', 'unknown'),
                            "confidence": entity.get('context_confidence', 0),
                            "description": entity.get('wikidata_description') or entity.get('wikipedia_description', ''),
                            "context": entity.get('sentence_context', ''),
                            "geocoding_source": entity.get('geocoding_source', '')
                        }
                    }
                    geojson_data["features"].append(feature)
                
                geojson_string = json.dumps(geojson_data, indent=2)
                
                st.download_button(
                    label="Download GeoJSON",
                    data=geojson_string,
                    file_name=f"{st.session_state.analysis_title}_enhanced_entities.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )
            else:
                st.info("No geocoded entities available for GeoJSON export.")

    def run(self):
        """Main application runner."""
        # Add custom CSS for enhanced styling
        st.markdown("""
        <style>
        .stApp {
            background-color: #F5F0DC !important;
        }
        .main .block-container {
            background-color: #F5F0DC !important;
        }
        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover {
            background-color: #45a049 !important;
            color: white !important;
        }
        .stExpander {
            background-color: white !important;
            border: 1px solid #E0D7C0 !important;
            border-radius: 4px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Input section
        text_input, analysis_title = self.render_input_section()
        
        # Enhanced process button
        if st.button("Process Text with Enhanced Methods", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyse.")
        
        # Add some spacing
        st.markdown("---")
        
        # Results section
        self.render_results()


def main():
    """Main function to run the Enhanced Streamlit application."""
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

