/**
 * Chat Interface Component for the Book Chatbot Widget
 */

export class ChatInterface {
    constructor(config) {
        this.config = config;
        this.messages = [];
        this.isLoading = false;
        this.currentRequestId = null;
    }

    async attachToDOM() {
        // Get DOM elements
        this.chatMessagesElement = document.getElementById('chat-messages');
        this.chatInputElement = document.getElementById('chat-input');

        if (!this.chatMessagesElement || !this.chatInputElement) {
            throw new Error('Chat interface elements not found in DOM');
        }

        // Load any existing session data
        await this.loadSession();
    }

    async sendMessage(message, options = {}) {
        if (this.isLoading) {
            console.warn('Message sending is in progress, ignoring new message');
            return;
        }

        // Add user message to UI
        this.addMessageToUI(message, 'user');

        // Set loading state
        this.isLoading = true;
        this.currentRequestId = Date.now(); // Simple ID for this request

        try {
            // Prepare the request payload
            const requestBody = {
                query: message,
                book_id: this.config.bookId,
                mode: options.mode || 'full-book'
            };

            // Add selected text if in selected-text mode
            if (options.mode === 'selected-text' && options.selectedText) {
                requestBody.selected_text = options.selectedText;
            }

            // Make API call to backend
            const response = await this.callApi('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.config.apiKey}`,
                    'X-API-Key': this.config.apiKey
                },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const data = await response.json();

                // Add bot response to UI
                this.addResponseToUI(data.response, data.citations);

                // Save to session
                await this.saveMessageToSession(message, data.response);
            } else {
                throw new Error(`API call failed with status: ${response.status}`);
            }
        } catch (error) {
            console.error('Error sending message:', error);

            // Add error message to UI
            this.addErrorMessageToUI('Sorry, I encountered an error processing your request. Please try again.');
        } finally {
            this.isLoading = false;
        }
    }

    async callApi(endpoint, options) {
        const url = `${this.config.apiUrl}${endpoint}`;
        return fetch(url, options);
    }

    addMessageToUI(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.className = `message message-${sender}`;

        messageElement.innerHTML = `
            <div class="message-content">${this.escapeHtml(message)}</div>
            <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
        `;

        this.chatMessagesElement.appendChild(messageElement);

        // Scroll to bottom
        this.chatMessagesElement.scrollTop = this.chatMessagesElement.scrollHeight;
    }

    addResponseToUI(response, citations) {
        // Add the response text
        this.addMessageToUI(response, 'bot');

        // If there are citations, add them separately
        if (citations && citations.length > 0) {
            const citationElement = document.createElement('div');
            citationElement.className = 'citations';

            citationElement.innerHTML = `
                <div class="citations-header">Sources:</div>
                ${citations.map(citation => `
                    <div class="citation">
                        <span class="citation-source">${this.escapeHtml(citation.source_text.substring(0, 100) + '...')}</span>
                        <div class="citation-location">
                            ${citation.location.chapter ? `Chapter: ${citation.location.chapter}` : ''}
                            ${citation.location.section ? ` | Section: ${citation.location.section}` : ''}
                            ${citation.location.page ? ` | Page: ${citation.location.page}` : ''}
                        </div>
                    </div>
                `).join('')}
            `;

            this.chatMessagesElement.appendChild(citationElement);
        }
    }

    addErrorMessageToUI(errorMessage) {
        const errorElement = document.createElement('div');
        errorElement.className = 'message message-error';

        errorElement.innerHTML = `
            <div class="message-content">${this.escapeHtml(errorMessage)}</div>
            <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
        `;

        this.chatMessagesElement.appendChild(errorElement);

        // Scroll to bottom
        this.chatMessagesElement.scrollTop = this.chatMessagesElement.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async saveMessageToSession(userMessage, botResponse) {
        if (this.config.sessionStorage) {
            try {
                const sessionId = this.getSessionId();
                const messagePair = {
                    id: Date.now(),
                    timestamp: new Date().toISOString(),
                    user: userMessage,
                    bot: botResponse
                };

                await this.config.sessionStorage.saveMessage(sessionId, messagePair);
            } catch (error) {
                console.error('Error saving message to session:', error);
            }
        }
    }

    async loadSession() {
        if (this.config.sessionStorage) {
            try {
                const sessionId = this.getSessionId();
                const messages = await this.config.sessionStorage.loadMessages(sessionId);

                for (const message of messages) {
                    this.addMessageToUI(message.user, 'user');
                    this.addMessageToUI(message.bot, 'bot');
                }
            } catch (error) {
                console.error('Error loading session:', error);
            }
        }
    }

    getSessionId() {
        // Create a session ID based on book ID and other factors
        return `session_${this.config.bookId}_${Math.floor(Date.now() / (24 * 60 * 60 * 1000))}`; // Daily session
    }
}