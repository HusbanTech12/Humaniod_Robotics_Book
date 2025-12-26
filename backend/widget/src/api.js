/**
 * API client for the Book Chatbot Widget
 * Handles all communication with the backend API
 */

class WidgetApiClient {
    constructor(baseURL, apiKey) {
        this.baseURL = baseURL;
        this.apiKey = apiKey;
    }

    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultHeaders = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`,
            'X-API-Key': this.apiKey
        };

        const config = {
            ...options,
            headers: {
                ...defaultHeaders,
                ...options.headers
            }
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }

    async sendMessage(query, bookId, mode = 'full-book', selectedText = null) {
        const requestBody = {
            query,
            book_id: bookId,
            mode
        };

        if (mode === 'selected-text' && selectedText) {
            requestBody.selected_text = selectedText;
        }

        return this.makeRequest('/chat', {
            method: 'POST',
            body: JSON.stringify(requestBody)
        });
    }

    async getHealth() {
        return this.makeRequest('/health', {
            method: 'GET'
        });
    }

    async validateApiKey() {
        try {
            const healthData = await this.getHealth();
            return healthData.status === 'healthy';
        } catch (error) {
            console.error('API key validation failed:', error);
            return false;
        }
    }
}

// Export the API client
export { WidgetApiClient };