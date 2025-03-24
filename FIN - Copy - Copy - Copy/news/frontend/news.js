const API_URL = 'http://localhost:5000/news-feed';
let lastUpdateTime = null;
let currentCategory = 'all';

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function createFeaturedNews(article) {
    return `
        <div class="card featured-news">
            <div class="row g-0">
                <div class="col-md-6">
                    <img src="${article.urlToImage || 'https://via.placeholder.com/600x400?text=No+Image'}" 
                         class="img-fluid rounded-start" 
                         alt="${article.title}">
                </div>
                <div class="col-md-6">
                    <div class="card-body">
                        <h2 class="card-title">${article.title}</h2>
                        <p class="card-text">${article.description}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="news-source">${article.source}</span>
                            <span class="news-date">${formatDate(article.publishedAt)}</span>
                        </div>
                        <a href="${article.url}" target="_blank" class="btn btn-primary mt-3">Read More</a>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function createNewsListItem(article) {
    return `
        <div class="card mb-3 news-item ${article.sentiment}">
            <div class="row g-0">
                <div class="col-md-4">
                    <img src="${article.urlToImage || 'https://via.placeholder.com/300x200?text=No+Image'}" 
                         class="img-fluid rounded-start" 
                         alt="${article.title}">
                </div>
                <div class="col-md-8">
                    <div class="card-body">
                        <h5 class="card-title">${article.title}</h5>
                        <p class="card-text">${article.description}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="news-source">${article.source}</span>
                            <span class="news-date">${formatDate(article.publishedAt)}</span>
                        </div>
                        <a href="${article.url}" target="_blank" class="btn btn-outline-primary mt-2">Read More</a>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function updateSourcesList(articles) {
    const sources = {};
    articles.forEach(article => {
        sources[article.source] = (sources[article.source] || 0) + 1;
    });

    const sourcesList = document.getElementById('sources-list');
    sourcesList.innerHTML = Object.entries(sources)
        .sort(([,a], [,b]) => b - a)
        .map(([source, count]) => `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span>${source}</span>
                <span class="badge bg-primary">${count}</span>
            </div>
        `).join('');
}

function filterNewsByCategory(articles, category) {
    if (category === 'all') return articles;
    return articles.filter(article => article.sentiment === category);
}

function updateNewsPage(articles) {
    // Update featured news (first article)
    const featuredNews = document.getElementById('featured-news');
    if (articles.length > 0) {
        featuredNews.innerHTML = createFeaturedNews(articles[0]);
    }

    // Update news list
    const filteredArticles = filterNewsByCategory(articles, currentCategory);
    const newsList = document.getElementById('news-list');
    newsList.innerHTML = filteredArticles.slice(1).map(createNewsListItem).join('');

    // Update sources list
    updateSourcesList(articles);

    // Update metadata
    document.getElementById('article-count').textContent = articles.length;
    lastUpdateTime = new Date();
    document.getElementById('last-update').textContent = formatDate(lastUpdateTime);
}

// Add event listeners for category buttons
document.querySelectorAll('.list-group-item').forEach(button => {
    button.addEventListener('click', (e) => {
        // Update active state
        document.querySelectorAll('.list-group-item').forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
        
        // Update category and refresh news
        currentCategory = e.target.dataset.category;
        fetchNews();
    });
});

async function fetchNews() {
    try {
        const response = await fetch(API_URL);
        const data = await response.json();
        
        if (data.status === 'success') {
            updateNewsPage(data.articles);
        }
    } catch (error) {
        console.error('Error fetching news:', error);
    }
}

// Initial fetch
fetchNews();

// Update every 5 minutes
setInterval(fetchNews, 5 * 60 * 1000); 