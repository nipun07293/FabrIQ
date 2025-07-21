// ==============================================================================
// Global Variables & Configuration
// ==============================================================================
let allProducts = [];
let currentFilters = {};
const API_BASE = 'http://localhost:5000';

const FILTER_CONFIG = {
    category:     { id: 'categoryFilters',   key: 'category_name' },
    item:         { id: 'itemFilters',       key: 'item' },
    brand:        { id: 'brandFilters',      key: 'clip_brand' },
    looks:        { id: 'looksFilters',      key: 'looks' },
    colors:       { id: 'colorsFilters',     key: 'colors' },
    material:     { id: 'materialFilters',   key: 'prints' },
    sleeveLength: { id: 'sleeveFilters',     key: 'sleeveLength' },
    neckLine:     { id: 'neckFilters',       key: 'neckLine' },
    fit:          { id: 'fitFilters',        key: 'fit' },
    length:       { id: 'lengthFilters',     key: 'length' },
    textures:     { id: 'texturesFilters',   key: 'textures' },
    shape:        { id: 'shapeFilters',      key: 'shape' }
};

// ==============================================================================
// DOM Element Caching
// ==============================================================================
const loadingIndicator = document.getElementById('loadingIndicator');
const mainContent = document.getElementById('mainContent');
const productsGrid = document.getElementById('productsGrid');
const resultsCountEl = document.getElementById('resultsCount');
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const backToAllBtn = document.getElementById('backToAll');
const clearFiltersBtn = document.getElementById('clearFiltersBtn');
const imageUploadInput = document.getElementById('imageUploadInput');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const imagePreview = document.getElementById('imagePreview');
const cancelUploadBtn = document.getElementById('cancelUploadBtn');
const uploadContainer = document.getElementById('uploadContainer');

// ==============================================================================
// Initialization & Data Fetching
// ==============================================================================

async function initializeApp() {
    try {
        const [productsResponse, filtersResponse] = await Promise.all([
            fetch(`${API_BASE}/api/products`),
            fetch(`${API_BASE}/api/filters`)
        ]);

        if (!productsResponse.ok || !filtersResponse.ok) {
            throw new Error('Failed to fetch initial data from the server.');
        }
        
        const productsData = await productsResponse.json();
        const filtersData = await filtersResponse.json();
        
        if (productsData.success) {
            allProducts = productsData.data;
        } else {
            throw new Error(productsData.error || 'Could not load products.');
        }

        if (filtersData.success) {
            populateFilters(filtersData.filters);
        } else {
            throw new Error(filtersData.error || 'Could not load filters.');
        }
        
        displayProducts(allProducts);
        updateResultsCount(allProducts.length, allProducts.length, 'all');
        
        loadingIndicator.style.display = 'none';
        mainContent.style.display = 'grid';

    } catch (error) {
        showError('Could not connect to the backend. Please ensure the Flask server is running correctly. <br><br>Error: ' + error.message);
    }
}

// ==============================================================================
// Filtering Logic
// ==============================================================================

function initializeFilters() {
    currentFilters = Object.keys(FILTER_CONFIG).reduce((acc, key) => ({ ...acc, [key]: [] }), {});
}

function populateFilters(filters) {
    for (const filterType in FILTER_CONFIG) {
        const config = FILTER_CONFIG[filterType];
        const container = document.getElementById(config.id);
        if (container && filters[filterType]) {
            container.innerHTML = ''; // Clear previous
            filters[filterType].forEach(value => {
                const chip = document.createElement('div');
                chip.className = 'filter-chip';
                chip.textContent = value;
                chip.dataset.filterType = filterType;
                chip.dataset.value = value;
                chip.onclick = () => toggleFilter(filterType, value, chip);
                container.appendChild(chip);
            });
        }
    }
}

function toggleFilter(filterType, value, chipElement) {
    chipElement.classList.toggle('active');
    const activeChips = document.querySelectorAll(`.filter-chip[data-filter-type="${filterType}"].active`);
    currentFilters[filterType] = Array.from(activeChips).map(chip => chip.dataset.value);
    applyFilters();
}

function applyFilters() {
    let filteredProducts = allProducts.filter(product => {
        return Object.entries(currentFilters).every(([filterType, selectedValues]) => {
            if (selectedValues.length === 0) return true;
            const productValue = product[FILTER_CONFIG[filterType].key];
            return productValue ? selectedValues.includes(productValue) : false;
        });
    });

    displayProducts(filteredProducts);
    updateResultsCount(filteredProducts.length, allProducts.length, 'filter');
    backToAllBtn.style.display = 'none';
    resetUploader();
}

function clearAllFilters() {
    initializeFilters();
    document.querySelectorAll('.filter-chip.active').forEach(chip => chip.classList.remove('active'));
    displayProducts(allProducts);
    updateResultsCount(allProducts.length, allProducts.length, 'all');
    backToAllBtn.style.display = 'none';
    searchInput.value = '';
    resetUploader();
}

// ==============================================================================
// Recommendation Logic
// ==============================================================================

async function handleRecommendation(type, query) {
    if (!query) return;
    productsGrid.innerHTML = '<div class="loading">🔍 Finding similar styles for you...</div>';
    updateResultsCount(0, 0, 'recommend');
    backToAllBtn.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE}/api/recommend?type=${type}&query=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Server responded with an error.');
        
        const recsData = await response.json();
        if (recsData.success) {
            displayProducts(recsData.data);
            updateResultsCount(recsData.data.length, 0, 'recommend');
        } else {
            throw new Error(recsData.error || 'Could not load recommendations.');
        }
    } catch (error) {
        productsGrid.innerHTML = `<div class="error"><p>Could not fetch recommendations. ${error.message}</p></div>`;
    }
}

async function handleUploadRecommendation(file) {
    productsGrid.innerHTML = '<div class="loading">✨ Analyzing your image...</div>';
    updateResultsCount(0, 0, 'recommend');
    backToAllBtn.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/recommend_by_upload`, { method: 'POST', body: formData });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({error: 'Server returned an invalid response.'}));
            throw new Error(errorData.error || 'Server error during upload.');
        }

        const recsData = await response.json();
        if (recsData.success) {
            displayProducts(recsData.data);
            updateResultsCount(recsData.data.length, 0, 'recommend');
        } else {
            throw new Error(recsData.error || 'Could not load recommendations.');
        }
    } catch (error) {
        productsGrid.innerHTML = `<div class="error"><p>Upload failed. ${error.message}</p></div>`;
    }
}

// ==============================================================================
// UI Rendering & Helpers
// ==============================================================================

function displayProducts(products) {
    productsGrid.innerHTML = '';
    
    if (!products || products.length === 0) {
        productsGrid.innerHTML = '<div class="no-results">No products found matching your criteria.</div>';
        return;
    }
    
    const productCards = products.map(product => {
        // Helper to safely get properties with fallbacks for null/undefined values
        const getProp = (key, fallback) => (product && product[key] != null) ? product[key] : fallback;

        const fileName = getProp('file_name', '');
        const imageUrl = fileName ? `${API_BASE}/images/${fileName}` : 'https://placehold.co/300x400/f0f0f0/ccc?text=No+Image';
        
        const otherDetails = [
            getProp('sleeveLength', null), getProp('fit', null), getProp('prints', null), getProp('neckLine', null)
        ].filter(d => d).join(' &middot; '); // Filter out nulls before joining

        return `
        <div class="product-card">
            <div class="product-image-container">
                <img src="${imageUrl}" alt="${getProp('item', 'Image')}" class="product-image" loading="lazy" onerror="this.onerror=null;this.src='https://placehold.co/300x400/f0f0f0/ccc?text=Image+Missing';">
            </div>
            <div class="product-info">
                <div class="product-brand">${getProp('clip_brand', 'Unknown Brand')}</div>
                <div class="product-name">${getProp('item', 'Fashion Outfit')}</div>
                <div class="product-primary-detail">${getProp('looks', '')}</div>
                <div class="product-extra-details">${otherDetails || '&nbsp;'}</div>
            </div>
            <div class="product-actions">
                <button class="similar-btn" data-filename="${fileName}" ${!fileName ? 'disabled' : ''}>Find Similar</button>
            </div>
        </div>`;
    }).join('');
    
    productsGrid.innerHTML = productCards;
}

function updateResultsCount(count, total, mode) {
    if (mode === 'recommend') {
        resultsCountEl.textContent = `Showing ${count} recommendations`;
    } else if (mode === 'filter') {
        resultsCountEl.textContent = `Showing ${count} of ${total} products`;
    } else { // 'all'
        resultsCountEl.textContent = `Showing all ${total} products`;
    }
}

function resetUploader() {
    imageUploadInput.value = '';
    imagePreviewContainer.style.display = 'none';
    uploadContainer.querySelector('.upload-btn-wrapper').style.display = 'inline-block';
}

function showError(message) {
    mainContent.style.display = 'none';
    loadingIndicator.style.display = 'block';
    loadingIndicator.className = 'error';
    loadingIndicator.innerHTML = `<h3>Oops! Something went wrong.</h3><p>${message}</p>`;
}

// ==============================================================================
// Event Listeners
// ==============================================================================
function handleTextSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    clearAllFilters();
    handleRecommendation('text', query);
}

function handleImageSearch(event) {
    const target = event.target;
    if (target.classList.contains('similar-btn')) {
        const filename = target.dataset.filename;
        if (filename) {
            clearAllFilters();
            searchInput.value = `Recommendations for ${filename}`;
            handleRecommendation('image', filename);
        }
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);
clearFiltersBtn.addEventListener('click', clearAllFilters);
searchButton.addEventListener('click', handleTextSearch);
searchInput.addEventListener('keyup', (event) => { if (event.key === 'Enter') handleTextSearch(); });
productsGrid.addEventListener('click', handleImageSearch);
backToAllBtn.addEventListener('click', clearAllFilters);

imageUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => { imagePreview.src = e.target.result; };
        reader.readAsDataURL(file);

        imagePreviewContainer.style.display = 'flex';
        uploadContainer.querySelector('.upload-btn-wrapper').style.display = 'none';
        
        clearAllFilters();
        searchInput.value = `Recommendations for uploaded image`;
        handleUploadRecommendation(file);
    }
});

cancelUploadBtn.addEventListener('click', clearAllFilters);