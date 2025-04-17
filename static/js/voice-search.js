/**
 * Voice Search Integration
 * 
 * This module provides voice search capabilities using the Web Speech API.
 * It integrates with the existing search and autocomplete systems.
 */

class VoiceSearch {
    constructor(options = {}) {
        this.targetInputId = options.targetInputId || 'searchInput';
        this.buttonId = options.buttonId || 'voiceSearchButton';
        this.formId = options.formId || 'searchForm';
        this.autoSubmit = options.autoSubmit || false;
        this.language = options.language || 'en-US';
        this.continuousListening = options.continuousListening || false;
        this.interimResults = options.interimResults || true;
        this.maxDuration = options.maxDuration || 10000; // 10 seconds
        this.minConfidence = options.minConfidence || 0.5;
        this.customCommands = options.customCommands || {};
        this.onStart = options.onStart || (() => {});
        this.onResult = options.onResult || (() => {});
        this.onEnd = options.onEnd || (() => {});
        this.onError = options.onError || (() => {});
        this.debug = options.debug || true; // Enable debug by default
        
        this.isListening = false;
        this.timer = null;
        
        // Initialize SpeechRecognition
        this.initSpeechRecognition();
        this.attachEventListeners();
        
        // Debug info
        if (this.debug) {
            console.log(`Voice search initialized for #${this.targetInputId} with button #${this.buttonId}`);
            this._debugElement();
        }
    }
    
    _debugElement() {
        // Check if target elements exist
        const inputExists = document.getElementById(this.targetInputId) !== null;
        const buttonExists = document.getElementById(this.buttonId) !== null;
        
        console.log(`Target input #${this.targetInputId} exists: ${inputExists}`);
        console.log(`Button #${this.buttonId} exists: ${buttonExists}`);
        
        if (!inputExists) {
            console.error(`Cannot find input element with ID: ${this.targetInputId}`);
        }
        
        if (!buttonExists) {
            console.error(`Cannot find button element with ID: ${this.buttonId}`);
        }
    }
    
    initSpeechRecognition() {
        // Check browser support with better error logging
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Voice search is not supported in this browser.');
            console.warn('Browser info:', navigator.userAgent);
            this.handleBrowserNotSupported();
            return;
        }
        
        try {
            // Initialize speech recognition with explicit error handling
            this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            this.recognition.lang = this.language;
            this.recognition.continuous = this.continuousListening;
            this.recognition.interimResults = this.interimResults;
            
            // Set up event handlers with bind to ensure proper 'this' context
            this.recognition.onstart = this.handleStart.bind(this);
            this.recognition.onresult = this.handleResult.bind(this);
            this.recognition.onend = this.handleEnd.bind(this);
            this.recognition.onerror = this.handleError.bind(this);
            
            if (this.debug) {
                console.log('Speech recognition initialized successfully');
            }
        } catch (error) {
            console.error('Error initializing speech recognition:', error);
            this.handleBrowserNotSupported();
        }
    }
    
    attachEventListeners() {
        // Add click event to voice search button
        const button = document.getElementById(this.buttonId);
        if (button) {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                if (this.debug) {
                    console.log('Voice search button clicked');
                }
                this.toggleVoiceSearch();
            });
            
            if (this.debug) {
                console.log('Button event listener attached successfully');
            }
        } else {
            console.error(`Button with ID ${this.buttonId} not found`);
        }
    }
    
    toggleVoiceSearch() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }
    
    startListening() {
        try {
            // Get permission explicitly
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(() => {
                        if (this.debug) {
                            console.log('Microphone permission granted');
                        }
                        this._startRecognition();
                    })
                    .catch((error) => {
                        console.error('Microphone permission denied:', error);
                        this.handleError({ error: 'not-allowed' });
                    });
            } else {
                // Fall back to direct recognition if mediaDevices not available
                this._startRecognition();
            }
        } catch (error) {
            console.error('Error starting voice recognition:', error);
            this.handleError({ error });
        }
    }
    
    _startRecognition() {
        try {
            this.recognition.start();
            this.isListening = true;
            
            if (this.debug) {
                console.log('Recognition started');
            }
            
            // Set timeout to stop listening after maxDuration
            if (this.maxDuration > 0) {
                this.timer = setTimeout(() => {
                    if (this.isListening) {
                        if (this.debug) {
                            console.log('Max duration reached, stopping recognition');
                        }
                        this.stopListening();
                    }
                }, this.maxDuration);
            }
        } catch (error) {
            console.error('Error in _startRecognition:', error);
            this.handleError({ error });
        }
    }
    
    stopListening() {
        try {
            this.recognition.stop();
            this.isListening = false;
            
            if (this.debug) {
                console.log('Recognition stopped');
            }
            
            // Clear the timeout
            if (this.timer) {
                clearTimeout(this.timer);
                this.timer = null;
            }
        } catch (error) {
            console.error('Error stopping voice recognition:', error);
        }
    }
    
    handleStart() {
        console.log('Voice recognition started');
        
        // Update UI to show listening state
        const button = document.getElementById(this.buttonId);
        if (button) {
            button.classList.add('listening');
            
            // If button has an icon, update it
            const icon = button.querySelector('i');
            if (icon) {
                icon.className = icon.className.replace('fa-microphone', 'fa-microphone-slash');
            }
        }
        
        // Show recognition status to user
        this.showRecognitionStatus('Listening...');
        
        // Call the onStart callback
        this.onStart();
    }
    
    handleResult(event) {
        // Debug the raw event first
        if (this.debug) {
            console.log('Recognition result event:', event);
        }
        
        try {
            const results = event.results;
            if (!results || results.length === 0) {
                console.warn('No results in the event');
                return;
            }
            
            const currentResult = results[results.length - 1];
            if (!currentResult || currentResult.length === 0) {
                console.warn('Empty result in the event');
                return;
            }
            
            const transcript = currentResult[0].transcript.trim();
            const confidence = currentResult[0].confidence;
            const isFinal = currentResult.isFinal;
            
            console.log(`Voice recognition result: "${transcript}" (${confidence.toFixed(2)})`);
            
            // Only process results with acceptable confidence
            if (confidence >= this.minConfidence) {
                // Check for custom commands
                const isCommand = this.processCommand(transcript.toLowerCase());
                
                if (!isCommand) {
                    // Update the search input with the transcript
                    const input = document.getElementById(this.targetInputId);
                    if (input) {
                        input.value = transcript;
                        
                        // Trigger input event to activate autocomplete
                        const inputEvent = new Event('input', { bubbles: true });
                        input.dispatchEvent(inputEvent);
                        
                        if (this.debug) {
                            console.log(`Updated input #${this.targetInputId} with: "${transcript}"`);
                        }
                    } else {
                        console.error(`Input element #${this.targetInputId} not found`);
                    }
                    
                    // If final result and autoSubmit is enabled, submit the form
                    if (isFinal && this.autoSubmit) {
                        const form = document.getElementById(this.formId);
                        if (form) {
                            if (this.debug) {
                                console.log(`Auto-submitting form #${this.formId}`);
                            }
                            form.submit();
                        } else {
                            console.error(`Form #${this.formId} not found`);
                        }
                    }
                }
            } else {
                if (this.debug) {
                    console.log(`Ignoring low confidence result: ${confidence} < ${this.minConfidence}`);
                }
            }
            
            // Show recognition status to user
            if (!isFinal) {
                this.showRecognitionStatus(`Heard: "${transcript}"`);
            } else {
                this.showRecognitionStatus(`Recognized: "${transcript}"`);
            }
            
            // Call the onResult callback
            this.onResult(transcript, confidence, isFinal);
        } catch (error) {
            console.error('Error in handleResult:', error);
        }
    }
    
    handleEnd() {
        console.log('Voice recognition ended');
        
        // Update UI to show not listening state
        const button = document.getElementById(this.buttonId);
        if (button) {
            button.classList.remove('listening');
            
            // If button has an icon, update it
            const icon = button.querySelector('i');
            if (icon) {
                icon.className = icon.className.replace('fa-microphone-slash', 'fa-microphone');
            }
        }
        
        // Hide recognition status
        this.hideRecognitionStatus();
        
        // Update state
        this.isListening = false;
        
        // Call the onEnd callback
        this.onEnd();
    }
    
    handleError(event) {
        console.error('Voice recognition error:', event.error);
        
        // Update UI to show error state
        const button = document.getElementById(this.buttonId);
        if (button) {
            button.classList.remove('listening');
            button.classList.add('error');
            
            // Remove error class after a delay
            setTimeout(() => {
                button.classList.remove('error');
            }, 2000);
        }
        
        // Show error message
        let errorMessage = 'Error with voice recognition';
        switch (event.error) {
            case 'no-speech':
                errorMessage = 'No speech detected';
                break;
            case 'aborted':
                errorMessage = 'Speech recognition aborted';
                break;
            case 'audio-capture':
                errorMessage = 'Could not capture audio';
                break;
            case 'network':
                errorMessage = 'Network error occurred';
                break;
            case 'not-allowed':
            case 'service-not-allowed':
                errorMessage = 'Microphone access denied';
                break;
            case 'bad-grammar':
                errorMessage = 'Grammar error in recognition';
                break;
            case 'language-not-supported':
                errorMessage = 'Language not supported';
                break;
        }
        
        // Show error status to user
        this.showRecognitionStatus(errorMessage, 'error');
        setTimeout(() => {
            this.hideRecognitionStatus();
        }, 3000);
        
        // Update state
        this.isListening = false;
        
        // Clear the timeout
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }
        
        // Call the onError callback
        this.onError(event);
    }
    
    processCommand(transcript) {
        // Basic navigation commands
        const navigationCommands = {
            'go to home': () => { window.location.href = '/'; },
            'go to homepage': () => { window.location.href = '/'; },
            'go to cart': () => { window.location.href = '/cart'; },
            'view my cart': () => { window.location.href = '/cart'; },
            'checkout': () => { window.location.href = '/checkout'; },
            'go to checkout': () => { window.location.href = '/checkout'; },
            'view my orders': () => { window.location.href = '/order-history'; },
            'go to profile': () => { window.location.href = '/profile'; },
            'log out': () => { window.location.href = '/logout'; },
            'sign out': () => { window.location.href = '/logout'; }
        };
        
        // Combine built-in commands with custom commands
        const allCommands = { ...navigationCommands, ...this.customCommands };
        
        // Check if the transcript matches any command
        for (const command in allCommands) {
            if (transcript.includes(command)) {
                console.log(`Executing command: ${command}`);
                allCommands[command]();
                return true;
            }
        }
        
        return false;
    }
    
    showRecognitionStatus(message, type = 'info') {
        // Create or get status element
        let statusElement = document.getElementById('voice-recognition-status');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.id = 'voice-recognition-status';
            statusElement.className = 'voice-recognition-status';
            document.body.appendChild(statusElement);
        }
        
        // Update the status message and type
        statusElement.textContent = message;
        statusElement.className = `voice-recognition-status ${type}`;
        statusElement.style.display = 'block';
    }
    
    hideRecognitionStatus() {
        const statusElement = document.getElementById('voice-recognition-status');
        if (statusElement) {
            statusElement.style.display = 'none';
        }
    }
    
    handleBrowserNotSupported() {
        // Hide or disable voice search button
        const button = document.getElementById(this.buttonId);
        if (button) {
            // Add not-supported class
            button.classList.add('not-supported');
            button.title = 'Voice search is not supported in this browser';
            
            // Also show an alert to inform the user
            this.showRecognitionStatus('Voice search is not supported in this browser. Please try Chrome or Edge.', 'error');
            setTimeout(() => {
                this.hideRecognitionStatus();
            }, 5000);
            
            if (this.debug) {
                console.warn(`Voice search button #${this.buttonId} marked as not supported`);
            }
        }
    }
}

// Add CSS styles for voice search
document.addEventListener('DOMContentLoaded', function() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .voice-search-button {
            cursor: pointer;
            transition: all 0.3s ease;
            color: #6c757d;
        }
        
        .voice-search-button:hover {
            color: #007bff;
        }
        
        .voice-search-button.listening {
            color: #dc3545;
            animation: pulse 1.5s infinite;
        }
        
        .voice-search-button.error {
            color: #dc3545;
        }
        
        .voice-search-button.not-supported {
            color: #6c757d;
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        .voice-recognition-status {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-radius: 20px;
            z-index: 1000;
            font-size: 14px;
            display: none;
        }
        
        .voice-recognition-status.error {
            background-color: rgba(220, 53, 69, 0.9);
        }
    `;
    document.head.appendChild(styleElement);
});

// Export the VoiceSearch class
window.VoiceSearch = VoiceSearch;