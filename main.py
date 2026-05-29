import streamlit as st
import numpy as np
from PIL import Image
from transformers import pipeline
import io
import time

# Set premium page configuration
st.set_page_config(
    page_title="Segmentador Inteligente de Folhas",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design and layout (harmonious dark/green theme)
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp [data-testid="stHeader"] {
        background-color: rgba(15, 23, 42, 0.8);
    }
    div[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #10b981 !important;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #059669;
        color: white;
    }
    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.markdown("### ⚙️ Configurações do Processamento")

model_selection = st.sidebar.selectbox(
    "Modelo de Segmentação",
    options=["SAM 1 (Base - Rápido no CPU)", "SAM 2.1 (Tiny - Lento no CPU)"],
    help="O SAM 1 (Base) é cerca de 10-20 vezes mais rápido em CPUs graças a otimizações de vetorização no PyTorch. O SAM 2.1 (Tiny) é menor e mais recente, mas sua inferência automática de máscaras ainda não está otimizada para CPU na biblioteca transformers."
)

# Cache the SAM model pipeline
@st.cache_resource
def load_sam(model_selection):
    model_id = "facebook/sam-vit-base" if "SAM 1" in model_selection else "facebook/sam2.1-hiera-tiny"
    with st.spinner(f"Carregando o modelo {model_selection} na memória... (Apenas uma vez)"):
        # Load the selected SAM model from Hugging Face
        return pipeline("mask-generation", model=model_id, device=-1)

# Initialize generator
generator = load_sam(model_selection)

# App Header
st.markdown("""
<div style='text-align: center; padding: 10px 0 30px 0;'>
    <h1>🌿 Segmentador Inteligente de Folhas</h1>
    <p style='color: #94a3b8; font-size: 1.2rem;'>
        Isolamento de folhas de alta precisão com Segment Anything Model (SAM) e exportação com fundo transparente.
    </p>
</div>
""", unsafe_allow_html=True)


# Slider to choose inference resolution
max_size = st.sidebar.slider(
    "Resolução Máxima de Inferência (px)",
    min_value=256,
    max_value=1024,
    value=512,
    step=128,
    help="Valores menores executam consideravelmente mais rápido no CPU (ex: ~5-10s para 512px)."
)

# Slider to choose probability threshold
mask_threshold = st.sidebar.slider(
    "Limiar de Tolerância da Máscara",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
    help="Ajuste o limiar se a folha estiver sendo cortada ou se partes do fundo estiverem inclusas."
)

st.sidebar.markdown("""
---
### 💡 Dicas de Uso:
1. Carregue uma foto nítida e bem iluminada da folha.
2. Utilize **512px** para uma edição rápida e interativa. Se quiser maior precisão nas bordas antes do download final, aumente para **1024px**.
3. O download gerará uma imagem **PNG com fundo transparente** ideal para alimentar classificadores especializados.
""")

# File Uploader Container
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Selecione a imagem da folha (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Create columns for main work area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Imagem Original")
        st.image(image, use_container_width=True)
    
    # Process the image
    with st.spinner("Processando imagem e detectando máscaras..."):
        t_start = time.time()
        
        # Resize image for fast inference
        ratio = min(max_size / image.width, max_size / image.height)
        if ratio < 1.0:
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image_resized = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            image_resized = image
        
        # Run SAM inference
        outputs = generator(image_resized)
        masks = outputs["masks"]
        
        t_duration = time.time() - t_start
        st.sidebar.success(f"⚡ Inferência concluída em {t_duration:.2f}s!")
    
    if len(masks) > 0:
        # Sort masks by size/pixel count descending
        masks_sorted = sorted(masks, key=lambda m: np.array(m).sum(), reverse=True)
        
        # Filter background-like masks (covering > 95% of image)
        valid_masks = []
        for m in masks_sorted:
            mask_np = np.array(m) > 0
            if mask_np.sum() < (0.95 * mask_np.size):
                valid_masks.append(mask_np)
        
        if not valid_masks:
            valid_masks = [np.array(m) > 0 for m in masks_sorted]
        
        st.sidebar.markdown(f"**Máscaras detectadas pelo SAM:** {len(valid_masks)}")
        
        # Let the user select the segment (default is the first/largest one)
        mask_idx = st.sidebar.selectbox(
            "Selecionar Máscara/Segmento da Folha",
            options=range(len(valid_masks)),
            format_func=lambda x: f"Segmento #{x+1} ({valid_masks[x].sum()} pixels)",
            index=0
        )
        
        selected_mask = valid_masks[mask_idx]
        
        # Convert resized image to numpy for masking
        img_np = np.array(image_resized)
        
        # Generate transparency (RGBA)
        # Create an alpha channel where mask is 255 (opaque) and background is 0 (transparent)
        alpha_channel = (selected_mask * 255).astype(np.uint8)
        rgba_img = np.dstack((img_np, alpha_channel))
        
        # Find bounding box coordinates to crop
        rows, cols = np.where(selected_mask)
        if len(rows) > 0 and len(cols) > 0:
            ymin, ymax = rows.min(), rows.max()
            xmin, xmax = cols.min(), cols.max()
            
            # Crop image
            cropped_rgba = rgba_img[ymin:ymax+1, xmin:xmax+1]
            cropped_pil = Image.fromarray(cropped_rgba)
            
            # Display results in col2
            with col2:
                st.subheader("✂️ Resultado do Recorte")
                
                # Option to show black background or checkers pattern (transparent)
                view_mode = st.radio("Fundo da visualização:", ["Transparente (Padrão)", "Contraste (Fundo Preto)"], horizontal=True)
                
                if view_mode == "Contraste (Fundo Preto)":
                    black_bg = np.zeros_like(img_np)
                    black_bg[selected_mask] = img_np[selected_mask]
                    cropped_black = black_bg[ymin:ymax+1, xmin:xmax+1]
                    st.image(cropped_black, use_container_width=True)
                else:
                    st.image(cropped_pil, use_container_width=True)
                
                # Save cropped image to memory buffer for download
                buf = io.BytesIO()
                cropped_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                # Download button
                st.download_button(
                    label="💾 Baixar Folha Recortada (PNG)",
                    data=byte_im,
                    file_name="folha_segmentada.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Statistics Card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 📊 Informações de Segmentação")
                leaf_area_pct = (selected_mask.sum() / selected_mask.size) * 100
                st.markdown(f"**Área da folha:** {selected_mask.sum()} pixels ({leaf_area_pct:.1f}% da imagem)")
                st.markdown(f"**Tamanho do recorte:** {cropped_pil.width}x{cropped_pil.height} px")
                st.markdown(f"**Bordas:** ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("A máscara selecionada está vazia.")
    else:
        st.error("Nenhum objeto pôde ser segmentado na imagem carregada.")
else:
    # Visual placeholder when no file is uploaded
    st.markdown("<div class='card' style='text-align: center; padding: 80px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3>📂 Faça o upload de uma imagem para começar</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>As imagens carregadas serão processadas localmente e não serão salvas.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
