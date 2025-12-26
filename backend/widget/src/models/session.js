/**
 * Session Model for the Book Chatbot Widget
 * Represents a client-side session with conversation history
 */

export class SessionModel {
    constructor(sessionId, bookId) {
        this.id = sessionId || this.generateSessionId();
        this.bookId = bookId;
        this.createdAt = new Date().toISOString();
        this.lastActivity = new Date().toISOString();
        this.messages = [];
        this.metadata = {};
    }

    generateSessionId() {
        // Generate a unique session ID
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    addMessage(role, content, timestamp = null) {
        const message = {
            id: this.generateMessageId(),
            role, // 'user' or 'assistant'
            content,
            timestamp: timestamp || new Date().toISOString()
        };

        this.messages.push(message);
        this.lastActivity = new Date().toISOString();

        return message;
    }

    generateMessageId() {
        // Generate a unique message ID
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getMessages() {
        return [...this.messages]; // Return a copy to prevent external modification
    }

    getUserMessages() {
        return this.messages.filter(msg => msg.role === 'user');
    }

    getAssistantMessages() {
        return this.messages.filter(msg => msg.role === 'assistant');
    }

    getMessageById(id) {
        return this.messages.find(msg => msg.id === id);
    }

    clearMessages() {
        this.messages = [];
        this.lastActivity = new Date().toISOString();
    }

    updateMetadata(key, value) {
        this.metadata[key] = value;
        this.lastActivity = new Date().toISOString();
    }

    getMetadata(key) {
        return this.metadata[key];
    }

    toJSON() {
        return {
            id: this.id,
            bookId: this.bookId,
            createdAt: this.createdAt,
            lastActivity: this.lastActivity,
            messages: this.messages,
            metadata: this.metadata
        };
    }

    static fromJSON(json) {
        const session = new SessionModel(json.id, json.bookId);
        session.createdAt = json.createdAt;
        session.lastActivity = json.lastActivity;
        session.messages = json.messages || [];
        session.metadata = json.metadata || {};

        return session;
    }

    isExpired(maxAgeHours = 24) {
        const lastActivity = new Date(this.lastActivity);
        const now = new Date();
        const ageInHours = (now - lastActivity) / (1000 * 60 * 60);

        return ageInHours > maxAgeHours;
    }

    getDuration() {
        const start = new Date(this.createdAt);
        const end = new Date(this.lastActivity);
        return end - start; // Duration in milliseconds
    }

    getWordCount() {
        return this.messages.reduce((total, message) => {
            return total + (message.content ? message.content.split(/\s+/).length : 0);
        }, 0);
    }

    getConversationTurns() {
        // A turn consists of a user message and a corresponding assistant message
        const userMessages = this.getUserMessages();
        const assistantMessages = this.getAssistantMessages();
        return Math.min(userMessages.length, assistantMessages.length);
    }
}