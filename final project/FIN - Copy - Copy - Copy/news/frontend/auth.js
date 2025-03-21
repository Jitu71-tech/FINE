const API_URL = 'http://localhost:5000';

// Simple user storage (in production, this would be handled by a backend)
const users = JSON.parse(localStorage.getItem('users') || '[]');

// Check if user is already logged in
function checkAuth() {
    const token = localStorage.getItem('token');
    const currentUser = localStorage.getItem('currentUser') || sessionStorage.getItem('currentUser');
    
    if (window.location.pathname.includes('login.html') && token && currentUser) {
        window.location.href = 'index.html';
    } else if (!window.location.pathname.includes('login.html') && (!token || !currentUser)) {
        window.location.href = 'login.html';
    }
}

// Handle Sign In
async function handleSignIn(event) {
    event.preventDefault();
    
    const email = document.getElementById('signin-email').value;
    const password = document.getElementById('signin-password').value;
    const rememberMe = document.getElementById('remember-me').checked;
    const submitButton = event.target.querySelector('button[type="submit"]');
    const originalButtonText = submitButton.innerHTML;

    try {
        // Disable submit button and show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Signing in...';

        const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();

        if (response.ok) {
            // Store token and user data
            localStorage.setItem('token', data.token);
            if (rememberMe) {
                localStorage.setItem('currentUser', JSON.stringify(data.user));
                sessionStorage.removeItem('currentUser');
            } else {
                sessionStorage.setItem('currentUser', JSON.stringify(data.user));
                localStorage.removeItem('currentUser');
            }
            showToast('Success', 'Login successful! Redirecting...', 'success');
            setTimeout(() => window.location.href = 'index.html', 1000);
        } else {
            showToast('Error', data.error || 'Login failed', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showToast('Error', 'An error occurred during login', 'error');
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = originalButtonText;
    }
    
    return false;
}

// Handle Sign Up
async function handleSignUp(event) {
    event.preventDefault();
    
    const name = document.getElementById('signup-name').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    const confirmPassword = document.getElementById('signup-confirm-password').value;
    const submitButton = event.target.querySelector('button[type="submit"]');
    const originalButtonText = submitButton.innerHTML;

    // Validation
    if (password !== confirmPassword) {
        showToast('Error', 'Passwords do not match', 'error');
        return false;
    }

    if (password.length < 6) {
        showToast('Error', 'Password must be at least 6 characters long', 'error');
        return false;
    }

    if (users.some(u => u.email === email)) {
        alert('Email already registered');
        return false;
    }

    try {
        // Disable submit button and show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Creating account...';

        const response = await fetch(`${API_URL}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, email, password })
        });

        const data = await response.json();

        if (response.ok) {
            // Store token and user data
            localStorage.setItem('token', data.token);
            localStorage.setItem('currentUser', JSON.stringify(data.user));
            showToast('Success', 'Account created successfully! Redirecting...', 'success');
            setTimeout(() => window.location.href = 'index.html', 1000);
        } else {
            showToast('Error', data.error || 'Registration failed', 'error');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showToast('Error', 'An error occurred during registration', 'error');
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = originalButtonText;
    }
    
    return false;
}

// Handle Logout
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('currentUser');
    sessionStorage.removeItem('currentUser');
    showToast('Success', 'Logged out successfully!', 'success');
    setTimeout(() => window.location.href = 'login.html', 1000);
}

// Toast notification function
function showToast(title, message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'position-fixed top-0 end-0 p-3';
        document.body.appendChild(container);
    }

    const toastId = 'toast-' + Date.now();
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : 'success'} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}</strong><br>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;

    document.getElementById('toast-container').insertAdjacentHTML('beforeend', toastHTML);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
    toast.show();

    // Remove toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Check authentication on page load
checkAuth(); 