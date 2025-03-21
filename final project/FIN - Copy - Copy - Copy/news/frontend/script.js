// Check if user is authenticated
function checkAuth() {
    const currentUser = localStorage.getItem('currentUser') || sessionStorage.getItem('currentUser');
    if (!currentUser) {
        window.location.href = 'login.html';
        return;
    }
    
    // Update user info in the navbar
    const user = JSON.parse(currentUser);
    document.getElementById('user-name').textContent = user.name;
}

const API_URL = 'http://localhost:5000/news-feed';
let lastUpdateTime = null;

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function updateSentimentStats(articles) {
    const stats = {
        positive: 0,
        neutral: 0,
        negative: 0
    };

    articles.forEach(article => {
        stats[article.sentiment]++;
    });

    document.getElementById('positive-count').textContent = stats.positive;
    document.getElementById('neutral-count').textContent = stats.neutral;
    document.getElementById('negative-count').textContent = stats.negative;
}

function createNewsCard(article) {
    return `
        <div class="col-md-6 col-lg-4">
            <div class="card news-card ${article.sentiment}">
                <img src="${article.urlToImage || 'https://via.placeholder.com/400x200?text=No+Image'}" 
                     class="card-img-top news-image" 
                     alt="${article.title}">
                <div class="card-body">
                    <h5 class="card-title">${article.title}</h5>
                    <p class="card-text">${article.description}</p>
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="news-source">${article.source}</span>
                        <span class="news-date">${formatDate(article.publishedAt)}</span>
                    </div>
                    <a href="${article.url}" target="_blank" class="btn btn-primary mt-2">Read More</a>
                </div>
            </div>
        </div>
    `;
}

function updateNewsFeed(articles) {
    const newsContainer = document.getElementById('news-container');
    newsContainer.innerHTML = articles.map(createNewsCard).join('');
    
    document.getElementById('article-count').textContent = articles.length;
    lastUpdateTime = new Date();
    document.getElementById('last-update').textContent = formatDate(lastUpdateTime);
    
    updateSentimentStats(articles);
}

async function fetchNews() {
    try {
        const response = await fetch(API_URL);
        const data = await response.json();
        
        if (data.status === 'success') {
            updateNewsFeed(data.articles);
        }
    } catch (error) {
        console.error('Error fetching news:', error);
    }
}

// Check authentication on page load
checkAuth();

// Initial fetch
fetchNews();

// Update every 5 minutes
setInterval(fetchNews, 5 * 60 * 1000);