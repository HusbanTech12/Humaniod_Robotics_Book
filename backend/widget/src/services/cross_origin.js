/**
 * Cross-Origin Communication Service for the Book Chatbot Widget
 * Handles communication between the widget and the parent page when embedded in an iframe
 */

export class CrossOriginService {
    constructor() {
        this.messageHandlers = {};
        this.parentOrigin = '*'; // Will be set when connecting to parent
        this.isConnected = false;

        // Start listening for messages
        this.listenForMessages();
    }

    /**
     * Listen for messages from the parent page
     */
    listenForMessages() {
        window.addEventListener('message', (event) => {
            // Set parent origin on first message for security
            if (!this.isConnected) {
                this.parentOrigin = event.origin;
                this.isConnected = true;
            }

            // Verify the origin if we're in a secure context
            if (this.isValidOrigin(event.origin)) {
                this.handleMessage(event.data);
            }
        });
    }

    /**
     * Check if the origin is valid for communication
     */
    isValidOrigin(origin) {
        // In a real implementation, you'd validate against allowed origins
        // For now, accept any origin (not recommended for production)
        return true;
    }

    /**
     * Handle incoming messages
     */
    handleMessage(data) {
        if (data.type && this.messageHandlers[data.type]) {
            this.messageHandlers[data.type].forEach(handler => {
                handler(data.payload);
            });
        }
    }

    /**
     * Send a message to the parent page
     */
    sendMessage(type, payload) {
        if (window.parent && this.isConnected) {
            window.parent.postMessage({
                type,
                payload,
                timestamp: Date.now()
            }, this.parentOrigin);
        }
    }

    /**
     * Register a handler for a specific message type
     */
    onMessageType(messageType, handler) {
        if (!this.messageHandlers[messageType]) {
            this.messageHandlers[messageType] = [];
        }
        this.messageHandlers[messageType].push(handler);
    }

    /**
     * Request data from the parent page
     */
    requestData(requestType, requestData = null) {
        return new Promise((resolve, reject) => {
            const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

            // Set up a temporary handler for the response
            const responseHandler = (response) => {
                if (response.requestId === requestId && response.type === 'response') {
                    resolve(response.payload);
                    // Remove this handler after use
                    const index = this.messageHandlers[`${requestType}_response`].indexOf(responseHandler);
                    if (index > -1) {
                        this.messageHandlers[`${requestType}_response`].splice(index, 1);
                    }
                }
            };

            if (!this.messageHandlers[`${requestType}_response`]) {
                this.messageHandlers[`${requestType}_response`] = [];
            }
            this.messageHandlers[`${requestType}_response`].push(responseHandler);

            // Send the request
            this.sendMessage('request', {
                type: requestType,
                payload: requestData,
                requestId
            });

            // Reject after timeout
            setTimeout(() => {
                reject(new Error(`Request ${requestType} timed out`));
                // Clean up handler
                const index = this.messageHandlers[`${requestType}_response`].indexOf(responseHandler);
                if (index > -1) {
                    this.messageHandlers[`${requestType}_response`].splice(index, 1);
                }
            }, 5000); // 5 second timeout
        });
    }

    /**
     * Initialize communication with parent page
     */
    async initializeCommunication() {
        try {
            // Send an initialization message to the parent
            this.sendMessage('initialized', {
                widgetType: 'book-chatbot',
                timestamp: Date.now()
            });

            // Request necessary configuration from parent
            const config = await this.requestData('getConfig');
            return config;
        } catch (error) {
            console.error('Error initializing communication with parent:', error);
            throw error;
        }
    }
}