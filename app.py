import streamlit as st
import json
import re
import hashlib
import os
import pandas as pd
from datetime import datetime
from gliner import GLiNER

# Set page configuration
st.set_page_config(
    page_title="Text Privacy Masker",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'masked_text' not in st.session_state:
    st.session_state.masked_text = ""
if 'unmasked_text' not in st.session_state:
    st.session_state.unmasked_text = ""
if 'display_masked' not in st.session_state:
    st.session_state.display_masked = True
if 'text_id' not in st.session_state:
    st.session_state.text_id = ""
if 'entity_map' not in st.session_state:
    st.session_state.entity_map = {}
if 'masking_stats' not in st.session_state:
    st.session_state.masking_stats = {}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'skipped_entities' not in st.session_state:
    st.session_state.skipped_entities = []

# Load GLiNER model
@st.cache_resource
def load_gliner_model():
    try:
        return GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
    except Exception as e:
        st.error(f"Error loading GLiNER model: {e}")
        return None

model = load_gliner_model()

# Predefined entity labels for GLiNER
ENTITY_LABELS = [
    "person names", 
    "company names", 
    "organisation", 
    "location", 
    "email", 
    "phone", 
    "url"
]

def get_mapping_filename(text_id):
    """Generate a unique filename for each text's mapping"""
    os.makedirs("mappings", exist_ok=True)
    return f"mappings/entity_mapping_{text_id}.json"

def mask_sensitive_information(text, text_id=None):
    """
    Masks sensitive information in text using GLiNER model.
    
    Args:
        text (str): The input text containing sensitive information
        text_id (str): Optional unique identifier for the text
        
    Returns:
        tuple: (masked_text, text_id, entity_map, masking_stats)
    """
    start_time = datetime.now()
    
    # Generate a unique ID if not provided
    if text_id is None:
        text_id = hashlib.md5(text.encode()).hexdigest()[:8]
    
    # Create a copy of the original text
    masked_text = text
    
    # Dictionary to track replacements
    entity_map = {}
    
    # Track stats for various entity types
    original_entities = {}
    skipped_entities = []
    
    # Track already masked spans to avoid overlaps
    masked_spans = []
    
    # Step 1: Use GLiNER to identify entities
    try:
        entities = model.predict_entities(text, ENTITY_LABELS)
        
        # Sort entities by length in descending order to prevent partial replacements
        entities_to_mask = []
        for entity in entities:
            entity_label = entity['label']
            entity_text = entity['text']
            start_char = entity['start']
            end_char = entity['end']
            confidence = entity['score']
            
            # Skip entities with very low confidence (can adjust threshold)
            if confidence < 0.45:
                skipped_entities.append((entity_text, entity_label, f"low_confidence: {confidence:.2f}"))
                continue
                
            entities_to_mask.append((entity_text, start_char, end_char, entity_label, confidence))
            
            if entity_label not in original_entities:
                original_entities[entity_label] = []
            original_entities[entity_label].append(entity_text)
        
        # Sort by length in descending order to prevent partial replacements
        entities_to_mask.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Mask named entities
        for ent_text, start_char, end_char, label, confidence in entities_to_mask:
            # Skip if this span overlaps with an already masked one
            if any(start_char < end and end_char > start for start, end in masked_spans):
                continue
                
            placeholder = f"[{label.upper()}_{len(entity_map) + 1}]"
            entity_map[placeholder] = ent_text
            
            # Replace all occurrences of this specific entity
            masked_text = masked_text.replace(ent_text, placeholder)
            
            # Mark as masked
            masked_spans.append((start_char, end_char))
    
    except Exception as e:
        st.error(f"Error processing text with GLiNER: {e}")
    
    # Step 2: Add regex-based detection for emails and URLs that might be missed by GLiNER
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_found = re.findall(email_pattern, masked_text)
    for i, email_addr in enumerate(emails_found):
        placeholder = f"[EMAIL_{i + 1}]"
        entity_map[placeholder] = email_addr
        masked_text = masked_text.replace(email_addr, placeholder)
        if "EMAIL" not in original_entities:
            original_entities["EMAIL"] = []
        original_entities["EMAIL"].append(email_addr)
    
    # URL pattern
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*\??[\w=&\-\.]*'
    urls_found = re.findall(url_pattern, masked_text)
    for i, url in enumerate(urls_found):
        placeholder = f"[URL_{i + 1}]"
        entity_map[placeholder] = url
        masked_text = masked_text.replace(url, placeholder)
        if "URL" not in original_entities:
            original_entities["URL"] = []
        original_entities["URL"].append(url)
    
    # Add additional phone detection for common formats not caught by GLiNER
    phone_patterns = [
        # Standard phone numbers with optional country code
        r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        # International format
        r'\+\d{1,3}\s?\d{1,14}',
        # Phone with extension
        r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(\s*ext\.?\s*\d{1,5})',
    ]

    for i, pattern in enumerate(phone_patterns):
        phone_matches = list(re.finditer(pattern, masked_text))
        for j, match in enumerate(phone_matches):
            phone = match.group(0)
            
            # Determine if this is a fax number
            is_fax = phone.lower().startswith("fax") or "fax" in phone.lower()
            placeholder_type = "FAX" if is_fax else "PHONE"
            
            placeholder = f"[{placeholder_type}_{i}_{j + 1}]"
            entity_map[placeholder] = phone
            masked_text = masked_text.replace(phone, placeholder)
            
            if placeholder_type not in original_entities:
                original_entities[placeholder_type] = []
            original_entities[placeholder_type].append(phone)
    
    # Save mapping with unique filename
    mapping_filename = get_mapping_filename(text_id)
    with open(mapping_filename, "w") as f:
        json.dump(entity_map, f, indent=4)
    
    # Calculate masking stats
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    total_chars = len(text)
    masked_chars = sum(1 for a, b in zip(text, masked_text) if a != b)
    masking_percentage = (masked_chars / total_chars) * 100 if total_chars > 0 else 0
    entity_counts = {k: len(v) for k, v in original_entities.items()}
    
    masking_stats = {
        "text_id": text_id,
        "processing_time_seconds": processing_time,
        "original_length": total_chars,
        "masked_length": len(masked_text),
        "masking_percentage": f"{masking_percentage:.1f}%",
        "entity_counts": entity_counts,
        "total_entities_masked": len(entity_map),
        "skipped_entities": len(skipped_entities)
    }
    
    # Store skipped entities for display
    st.session_state.skipped_entities = skipped_entities
    
    return masked_text, text_id, entity_map, masking_stats

def unmask_text(masked_text, text_id=None, entity_map=None):
    """
    Restore original text using the stored mapping.
    
    Args:
        masked_text: The masked text
        text_id: The unique identifier for the text
        entity_map: Optional entity map to use instead of loading from file
        
    Returns:
        The unmasked text and entity map
    """
    if entity_map is None and text_id is not None:
        mapping_filename = get_mapping_filename(text_id)
        try:
            with open(mapping_filename, "r") as f:
                entity_map = json.load(f)
        except FileNotFoundError:
            return f"Error: Mapping file '{mapping_filename}' not found.", None
    
    if entity_map is None:
        return "Error: No entity mapping provided or found.", None
    
    unmasked_text = masked_text
    
    # Sort placeholders by length in descending order to prevent partial replacements
    placeholders = sorted(entity_map.keys(), key=len, reverse=True)
    
    # Replace placeholders with actual values
    for placeholder in placeholders:
        unmasked_text = unmasked_text.replace(placeholder, entity_map[placeholder])
    
    return unmasked_text, entity_map

def render_entity_mapping_table(entity_map):
    """Convert entity mapping to a DataFrame for display"""
    if not entity_map:
        return pd.DataFrame()
    
    # Extract entity type from the placeholder
    data = []
    for placeholder, original in entity_map.items():
        entity_type = placeholder.split('_')[0].strip('[]')
        data.append({
            "Placeholder": placeholder,
            "Entity Type": entity_type,
            "Original Value": original,
            "Length": len(original)
        })
    
    df = pd.DataFrame(data)
    return df

def add_to_history(action, text_id, text_length):
    """Add an action to the history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "timestamp": timestamp,
        "action": action,
        "text_id": text_id,
        "text_length": text_length
    })

# Streamlit UI
st.title("🔒 Text Privacy Masking System (GLiNER-powered)")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Mask/Unmask Text", "Entity Mapping", "History", "Skipped Entities"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Input Text")
        input_text = st.text_area(
            "Enter text below:",
            height=300,
            placeholder="Paste your text content here...",
            key="text_input"
        )
    
    with col2:
        st.subheader("Actions")
        
        # Text ID input
        manual_id = st.text_input(
            "Optional Text ID",
            placeholder="Leave blank for auto-generated ID",
            help="Provide an ID to link masked/unmasked texts"
        )
        
        # Mask button
        if st.button("📝 Mask Text", use_container_width=True):
            if input_text:
                if model is None:
                    st.error("GLiNER model failed to load. Please check your installation.")
                else:
                    text_id = manual_id if manual_id else None
                    with st.spinner("Processing with GLiNER model..."):
                        masked_text, text_id, entity_map, masking_stats = mask_sensitive_information(input_text, text_id)
                    
                    st.session_state.masked_text = masked_text
                    st.session_state.unmasked_text = input_text  # Store original text
                    st.session_state.display_masked = True  # Set display state to masked
                    st.session_state.text_id = text_id
                    st.session_state.entity_map = entity_map
                    st.session_state.masking_stats = masking_stats
                    
                    add_to_history("Masked", text_id, len(input_text))
                    
                    st.success(f"Text masked successfully! ID: {text_id}")
            else:
                st.warning("Please enter text to mask.")
        
        # Unmask button
        if st.button("🔓 Unmask Text", use_container_width=True):
            if st.session_state.masked_text:
                unmasked_text, _ = unmask_text(
                    st.session_state.masked_text,
                    st.session_state.text_id,
                    st.session_state.entity_map
                )
                
                st.session_state.unmasked_text = unmasked_text
                st.session_state.display_masked = False  # Set display state to unmasked
                
                add_to_history("Unmasked", st.session_state.text_id, len(unmasked_text))
                
                st.success("Text unmasked successfully!")
            else:
                st.warning("No masked text available. Please mask text first.")
        
        # Clear button
        if st.button("🧹 Clear All", use_container_width=True):
            st.session_state.masked_text = ""
            st.session_state.unmasked_text = ""
            st.session_state.display_masked = True
            st.session_state.text_id = ""
            st.session_state.entity_map = {}
            st.session_state.masking_stats = {}
            st.session_state.skipped_entities = []
            st.success("All data cleared!")
    
    # Text Display with toggling between masked and unmasked
    if st.session_state.masked_text:
        # Determine what to display based on the display_masked flag
        display_title = "Masked Text" if st.session_state.display_masked else "Unmasked Text"
        display_content = st.session_state.masked_text if st.session_state.display_masked else st.session_state.unmasked_text
        
        st.subheader(display_title)
        
        # Using expander to display the content
        text_container = st.expander(f"{display_title} Content", expanded=True)
        with text_container:
            st.code(display_content, language=None)
        
        # Statistics about masking (only show when displaying masked text)
        if st.session_state.masking_stats and st.session_state.display_masked:
            stats = st.session_state.masking_stats
            
            st.subheader("Masking Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Processing Time", f"{stats['processing_time_seconds']:.3f}s")
            col2.metric("Original Length", stats['original_length'])
            col3.metric("Masked Length", stats['masked_length'])
            col4.metric("Masking Rate", stats['masking_percentage'])
            col5.metric("Entities Masked", stats['total_entities_masked'])
            
            # Entity type breakdown
            if 'entity_counts' in stats and stats['entity_counts']:
                st.subheader("Entities Masked by Type")
                entity_df = pd.DataFrame({
                    'Entity Type': list(stats['entity_counts'].keys()),
                    'Count': list(stats['entity_counts'].values())
                })
                fig_data = entity_df.set_index('Entity Type')
                st.bar_chart(fig_data)

with tab2:
    st.subheader("Entity Mapping Dictionary")
    
    if st.session_state.entity_map:
        # Create a visual representation of the mapping dictionary
        mapping_df = render_entity_mapping_table(st.session_state.entity_map)
        
        # Use dataframe with fixed height
        st.dataframe(
            mapping_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Option to download the mapping as JSON
        if st.download_button(
            label="📥 Download Mapping as JSON",
            data=json.dumps(st.session_state.entity_map, indent=4),
            file_name=f"entity_mapping_{st.session_state.text_id}.json",
            mime="application/json"
        ):
            st.success("Mapping downloaded!")
        
        # Show raw JSON
        with st.expander("View Raw JSON"):
            st.code(json.dumps(st.session_state.entity_map, indent=4), language="json")
    else:
        st.info("No entity mapping available. Please mask text first.")

with tab3:
    st.subheader("Operation History")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.experimental_rerun()
    else:
        st.info("No operation history available.")

with tab4:
    st.subheader("Skipped Entities")
    
    if st.session_state.skipped_entities:
        skipped_df = pd.DataFrame(st.session_state.skipped_entities, 
                                  columns=["Text", "Entity Type", "Reason"])
        st.dataframe(
            skipped_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No entities have been skipped yet.")

# Sidebar with information and settings
with st.sidebar:
    st.title("📋 App Info")
    st.info(
        """
        **Text Privacy Masker** helps you protect sensitive information in any text.
        
        Powered by GLiNER model, it masks these entity types:
        - Person names
        - Company names
        - Organization names
        - Locations
        - Email addresses
        - Phone numbers
        - Website URLs
        
        Automatically skips:
        - Low confidence predictions
        - Ambiguous entities
        """
    )
    
    # Option to customize entity labels
    with st.expander("Advanced Settings"):
        custom_labels = st.multiselect(
        "Select entity types to mask:", 
        options=["person names", "company names", "organisation", "location", "email", "phone", "url"],
        default=ENTITY_LABELS,
        help="Customize which entity types to detect and mask"
    )
        if custom_labels != ENTITY_LABELS:
            ENTITY_LABELS[:] = custom_labels
            st.success("Entity labels updated!")
    
    st.subheader("Current Session")
    if st.session_state.text_id:
        st.success(f"Working with text ID: {st.session_state.text_id}")
    else:
        st.warning("No text currently being processed")
    
    # Show count of masked entities by type
    if st.session_state.masking_stats and 'entity_counts' in st.session_state.masking_stats:
        st.subheader("Masked Entity Counts")
        for entity_type, count in st.session_state.masking_stats['entity_counts'].items():
            st.text(f"{entity_type}: {count}")
    
    # Footer
    st.markdown("---")
    st.caption("© 2025 Text Privacy Masker v1.0 (GLiNER Edition)")