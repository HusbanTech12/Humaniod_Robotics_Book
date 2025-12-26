/**
 * Mobile Support Component for the Book Chatbot Widget
 * Adds mobile-specific interactions and touch support
 */

export class MobileSupport {
    constructor() {
        this.isTouchDevice = this.detectTouchDevice();
        this.init();
    }

    detectTouchDevice() {
        // Check for touch capabilities
        return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    }

    init() {
        if (this.isTouchDevice) {
            this.enableTouchOptimizations();
        }
        this.enableResponsiveFeatures();
    }

    enableTouchOptimizations() {
        // Add touch-specific event listeners
        document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true });
        document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true });

        // Optimize for touch
        this.optimizeForTouch();
    }

    enableResponsiveFeatures() {
        // Add responsive features for all devices
        window.addEventListener('resize', this.handleResize.bind(this));
        this.applyInitialResponsiveStyles();
    }

    handleTouchStart(event) {
        // Handle initial touch
        this.touchStartTime = Date.now();
        this.touchStartCoords = {
            x: event.touches[0].clientX,
            y: event.touches[0].clientY
        };
    }

    handleTouchMove(event) {
        // Prevent scrolling when interacting with the chat widget
        if (this.isTouchEventInWidget(event)) {
            event.preventDefault();
        }
    }

    handleTouchEnd(event) {
        // Handle touch end
        const touchDuration = Date.now() - this.touchStartTime;
        const touchEndCoords = {
            x: event.changedTouches[0].clientX,
            y: event.changedTouches[0].clientY
        };

        // Detect if it was a tap (not a swipe)
        if (touchDuration < 300) { // Less than 300ms is considered a tap
            const distance = Math.sqrt(
                Math.pow(touchEndCoords.x - this.touchStartCoords.x, 2) +
                Math.pow(touchEndCoords.y - this.touchStartCoords.y, 2)
            );

            if (distance < 10) { // Less than 10px movement is considered a tap
                this.handleTap(event, touchEndCoords);
            }
        }
    }

    handleTap(event, coords) {
        // Handle tap events specifically for mobile
        const tappedElement = document.elementFromPoint(coords.x, coords.y);

        // If tapping on input, ensure it's properly focused
        if (tappedElement && (tappedElement.tagName === 'INPUT' || tappedElement.tagName === 'TEXTAREA')) {
            tappedElement.focus();
        }
    }

    isTouchEventInWidget(event) {
        // Check if the touch event occurred within the chat widget
        const widgetElement = document.querySelector('.book-chatbot-container');
        if (!widgetElement) return false;

        const rect = widgetElement.getBoundingClientRect();
        const touch = event.touches[0];

        return (
            touch.clientX >= rect.left &&
            touch.clientX <= rect.right &&
            touch.clientY >= rect.top &&
            touch.clientY <= rect.bottom
        );
    }

    optimizeForTouch() {
        // Add CSS classes for touch-friendly styles
        const style = document.createElement('style');
        style.textContent = `
            /* Touch-friendly styles */
            .book-chatbot-container {
                -webkit-touch-callout: none;
                -webkit-user-select: none;
                -moz-user-select: none;
                -ms-user-select: none;
                user-select: none;
            }

            .message {
                max-width: 90%; /* Wider on mobile */
            }

            .chat-input-container input {
                padding: 16px; /* Larger touch target */
                font-size: 16px; /* Better for mobile */
            }

            .chat-input-container button {
                padding: 16px 24px; /* Larger touch target */
                font-size: 16px;
            }

            /* Adjustments for mobile screen sizes */
            @media (max-width: 768px) {
                .book-chatbot-container {
                    height: 400px;
                    border-radius: 0;
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    z-index: 10000;
                }

                .chat-messages {
                    padding: 12px;
                }

                .chat-input-container {
                    padding: 12px;
                }
            }
        `;
        document.head.appendChild(style);
    }

    handleResize() {
        // Handle window resize for responsive adjustments
        this.applyResponsiveStyles();
    }

    applyInitialResponsiveStyles() {
        // Apply initial responsive styles based on device characteristics
        this.applyResponsiveStyles();
    }

    applyResponsiveStyles() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        const container = document.querySelector('.book-chatbot-container');

        if (!container) return;

        // Adjust widget size based on screen dimensions
        if (width < 768) { // Mobile
            container.style.height = `${height * 0.6}px`;
            container.style.width = '100vw';
            container.style.maxHeight = 'none';
        } else if (width < 1024) { // Tablet
            container.style.height = '500px';
            container.style.width = '90vw';
            container.style.maxWidth = '500px';
        } else { // Desktop
            container.style.height = '600px';
            container.style.width = '400px';
            container.style.maxWidth = '400px';
        }
    }

    // Method to enable/disable mobile optimizations
    setMobileMode(enabled) {
        if (enabled && this.isTouchDevice) {
            this.enableTouchOptimizations();
        } else {
            this.disableTouchOptimizations();
        }
    }

    disableTouchOptimizations() {
        // Remove touch event listeners
        document.removeEventListener('touchstart', this.handleTouchStart);
        document.removeEventListener('touchmove', this.handleTouchMove);
        document.removeEventListener('touchend', this.handleTouchEnd);
    }
}