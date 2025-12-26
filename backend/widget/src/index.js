/**
 * Main entry point for the RAG Chatbot Widget
 */

// Import required modules
import { ChatInterface } from './components/ChatInterface.js';
import { TextSelection } from './components/TextSelection.js';
import { SessionStorage } from './services/session_storage.js';

// Default configuration
const DEFAULT_CONFIG = {
    apiUrl: 'http://localhost:8000/v1',
    bookId: null,
    apiKey: null,
    containerId: 'book-chatbot',
    theme: 'default'
};

class BookChatWidget {
    constructor(config = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.chatInterface = null;
        this.textSelection = null;
        this.sessionStorage = null;

        this.initialize();
    }

    async initialize() {
        try {
            // Initialize session storage
            this.sessionStorage = new SessionStorage();

            // Initialize text selection handler
            this.textSelection = new TextSelection();

            // Initialize chat interface
            this.chatInterface = new ChatInterface({
                apiUrl: this.config.apiUrl,
                bookId: this.config.bookId,
                apiKey: this.config.apiKey,
                sessionStorage: this.sessionStorage
            });

            // Render the widget
            await this.render();

            // Set up event listeners
            this.setupEventListeners();

            console.log('BookChatWidget initialized successfully');
        } catch (error) {
            console.error('Error initializing BookChatWidget:', error);
        }
    }

    async render() {
        const container = document.getElementById(this.config.containerId);
        if (!container) {
            throw new Error(`Container element with id '${this.config.containerId}' not found`);
        }

        // Create the widget container
        const widgetElement = document.createElement('div');
        widgetElement.id = 'book-chatbot-container';
        widgetElement.className = 'book-chatbot-container';
        widgetElement.innerHTML = `
            <div class="chat-header">
                <h3>Book Assistant</h3>
            </div>
            <div id="chat-messages" class="chat-messages">
                <div class="welcome-message">
                    <p>Ask me anything about this book!</p>
                </div>
            </div>
            <div id="chat-input-container" class="chat-input-container">
                <input type="text" id="chat-input" placeholder="Ask a question about the book..." />
                <button id="chat-send-btn">Send</button>
            </div>
        `;

        container.appendChild(widgetElement);

        // Attach chat interface to the rendered elements
        await this.chatInterface.attachToDOM();
    }

    setupEventListeners() {
        // Listen for text selection events
        this.textSelection.on('selection', (selectionData) => {
            this.handleTextSelection(selectionData);
        });

        // Listen for send button click
        const sendButton = document.getElementById('chat-send-btn');
        const inputField = document.getElementById('chat-input');

        sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        inputField.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    async sendMessage() {
        const inputField = document.getElementById('chat-input');
        const message = inputField.value.trim();

        if (!message) {
            return;
        }

        // Clear input
        inputField.value = '';

        // Send message through chat interface
        await this.chatInterface.sendMessage(message);
    }

    async handleTextSelection(selectionData) {
        // Show context menu for selected text
        const contextMenu = document.createElement('div');
        contextMenu.className = 'context-menu';
        contextMenu.style.position = 'absolute';
        contextMenu.style.left = `${selectionData.x}px`;
        contextMenu.style.top = `${selectionData.y}px`;
        contextMenu.innerHTML = `
            <button id="ask-ai-btn">Ask AI</button>
        `;

        document.body.appendChild(contextMenu);

        // Handle "Ask AI" button click
        document.getElementById('ask-ai-btn').addEventListener('click', () => {
            const question = `Based on this text: "${selectionData.text}", ${selectionData.question || 'what can you tell me?'}`;
            this.chatInterface.sendMessage(question, {
                mode: 'selected-text',
                selectedText: selectionData.text
            });

            // Remove context menu
            document.body.removeChild(contextMenu);
        });

        // Remove context menu after timeout
        setTimeout(() => {
            if (document.body.contains(contextMenu)) {
                document.body.removeChild(contextMenu);
            }
        }, 5000);
    }

    static init(config) {
        return new BookChatWidget(config);
    }
}

// Export the widget class
export { BookChatWidget };

// If running in a browser environment, attach to global scope
if (typeof window !== 'undefined') {
    window.BookChatWidget = BookChatWidget;
}