/**
 * Embedding Script for the Book Chatbot Widget
 * Provides methods to embed the widget in different ways
 */

import { BookChatWidget } from './index.js';

class WidgetEmbedder {
    constructor() {
        this.widgets = {};
    }

    /**
     * Embed the widget using an iframe
     */
    embedAsIframe(config) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container element with id '${config.containerId}' not found`);
        }

        // Create iframe element
        const iframe = document.createElement('iframe');
        iframe.src = `${config.widgetUrl}?bookId=${encodeURIComponent(config.bookId)}&apiKey=${encodeURIComponent(config.apiKey)}`;
        iframe.width = '100%';
        iframe.height = config.height || '500px';
        iframe.frameBorder = '0';
        iframe.style.border = '1px solid #ccc';
        iframe.style.borderRadius = '8px';

        container.appendChild(iframe);

        return iframe;
    }

    /**
     * Embed the widget using a script tag approach
     */
    embedAsScript(config) {
        // Create the container div if it doesn't exist
        let container = document.getElementById(config.containerId);
        if (!container) {
            container = document.createElement('div');
            container.id = config.containerId;
            container.style.width = '100%';
            container.style.maxWidth = '500px';
            container.style.margin = '0 auto';
            document.body.appendChild(container);
        }

        // Initialize the widget
        const widget = BookChatWidget.init(config);
        this.widgets[config.containerId] = widget;

        return widget;
    }

    /**
     * Initialize the widget with default settings if not already done
     */
    static init(config) {
        // If no container ID is specified, create a default one
        if (!config.containerId) {
            config.containerId = 'book-chatbot-default';

            // Check if default container exists, if not create it
            let defaultContainer = document.getElementById(config.containerId);
            if (!defaultContainer) {
                defaultContainer = document.createElement('div');
                defaultContainer.id = config.containerId;
                defaultContainer.style.position = 'fixed';
                defaultContainer.style.bottom = '20px';
                defaultContainer.style.right = '20px';
                defaultContainer.style.zIndex = '10000';
                defaultContainer.style.width = '400px';
                defaultContainer.style.height = '600px';
                document.body.appendChild(defaultContainer);
            }
        }

        // Create an instance of WidgetEmbedder and embed the widget
        const embedder = new WidgetEmbedder();
        return embedder.embedAsScript(config);
    }
}

// Also make it available globally
if (typeof window !== 'undefined') {
    window.BookChatWidgetEmbedder = WidgetEmbedder;
}

// If this is included via a script tag and the init function is called globally
if (typeof window !== 'undefined' && window.bookChatConfig) {
    // Auto-initialize if configuration is provided globally
    WidgetEmbedder.init(window.bookChatConfig);
}

export { WidgetEmbedder };