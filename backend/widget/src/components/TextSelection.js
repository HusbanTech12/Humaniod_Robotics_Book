/**
 * Text Selection Component for the Book Chatbot Widget
 * Handles text selection events and provides selected text to other components
 */

export class TextSelection {
    constructor() {
        this.eventListeners = {};
        this.init();
    }

    init() {
        // Listen for mouseup event to detect text selection
        document.addEventListener('mouseup', this.handleMouseUp.bind(this));
    }

    handleMouseUp() {
        const selectedText = this.getSelectedText();

        if (selectedText) {
            // Get the coordinates for the context menu
            const selectionRange = this.getSelectionRange();
            if (selectionRange) {
                const rect = selectionRange.getBoundingClientRect();

                // Trigger the selection event with text and position
                this.trigger('selection', {
                    text: selectedText,
                    x: rect.left,
                    y: rect.top,
                    question: ''
                });
            }
        }
    }

    getSelectedText() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const selectedText = selection.toString().trim();
            return selectedText.length > 0 ? selectedText : null;
        }
        return null;
    }

    getSelectionRange() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            return selection.getRangeAt(0);
        }
        return null;
    }

    // Event system for communication
    on(event, callback) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(callback);
    }

    trigger(event, data) {
        if (this.eventListeners[event]) {
            this.eventListeners[event].forEach(callback => callback(data));
        }
    }
}