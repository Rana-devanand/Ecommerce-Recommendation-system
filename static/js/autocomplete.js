
class SearchAutocomplete {
    constructor(inputSelector, options = {}) {
        // Configuration with defaults
        this.config = {
            minLength: 2,            // Minimum characters before triggering autocomplete
            maxResults: 10,          // Maximum number of suggestions to show
            delay: 300,              // Delay in ms before triggering search (debounce)
            endpoint: '/api/autocomplete', // API endpoint for suggestions
            highlightMatches: true,  // Highlight matching text in suggestions
            onSelect: null,          // Callback when an item is selected
            containerClass: 'autocomplete-container',
            suggestionsClass: 'autocomplete-suggestions',
            suggestionClass: 'autocomplete-suggestion',
            activeClass: 'autocomplete-active',
            ...options
        };
        
        // DOM elements
        this.inputElement = document.querySelector(inputSelector);
        if (!this.inputElement) {
            console.error(`Input element not found: ${inputSelector}`);
            return;
        }
        
        // Create and insert container for suggestions
        this.container = document.createElement('div');
        this.container.className = this.config.containerClass;
        this.container.style.position = 'relative';
        this.inputElement.parentNode.insertBefore(this.container, this.inputElement.nextSibling);
        
        // Move input inside container
        this.container.appendChild(this.inputElement);
        
        // Create suggestions element
        this.suggestionsElement = document.createElement('ul');
        this.suggestionsElement.className = this.config.suggestionsClass;
        this.suggestionsElement.style.display = 'none';
        this.container.appendChild(this.suggestionsElement);
        
        // Internal state
        this.suggestions = [];
        this.selectedIndex = -1;
        this.timeoutId = null;
        
        // Bind event handlers
        this.inputElement.addEventListener('input', this.onInput.bind(this));
        this.inputElement.addEventListener('keydown', this.onKeyDown.bind(this));
        this.suggestionsElement.addEventListener('click', this.onSuggestionClick.bind(this));
        document.addEventListener('click', this.onDocumentClick.bind(this));
        
        // Apply initial styles
        this.applyStyles();
    }
    
    applyStyles() {
        // Container styles
        Object.assign(this.container.style, {
            position: 'relative',
            width: '100%'
        });
        
        // Suggestions list styles
        Object.assign(this.suggestionsElement.style, {
            position: 'absolute',
            top: '100%',
            left: '0',
            zIndex: '1000',
            width: '100%',
            maxHeight: '300px',
            overflowY: 'auto',
            backgroundColor: '#fff',
            border: '1px solid #ddd',
            borderTop: 'none',
            borderRadius: '0 0 4px 4px',
            boxShadow: '0 6px 12px rgba(0,0,0,0.175)',
            padding: '0',
            margin: '0',
            listStyle: 'none'
        });
    }
    
    onInput() {
        const query = this.inputElement.value.trim();
        
        // Clear any existing timeout
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
        }
        
        // Hide suggestions if query is too short
        if (query.length < this.config.minLength) {
            this.hideSuggestions();
            return;
        }
        
        // Debounce the API call
        this.timeoutId = setTimeout(() => {
            this.fetchSuggestions(query);
        }, this.config.delay);
    }
    
    onKeyDown(event) {
        // If suggestions are not visible, don't handle navigation keys
        if (this.suggestionsElement.style.display === 'none') {
            return;
        }
        
        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                this.selectNext();
                break;
            case 'ArrowUp':
                event.preventDefault();
                this.selectPrevious();
                break;
            case 'Enter':
                if (this.selectedIndex >= 0) {
                    event.preventDefault();
                    this.selectSuggestion(this.selectedIndex);
                }
                break;
            case 'Escape':
                this.hideSuggestions();
                break;
        }
    }
    
    onSuggestionClick(event) {
        const item = event.target.closest(`.${this.config.suggestionClass}`);
        if (item) {
            const index = Array.from(this.suggestionsElement.children).indexOf(item);
            this.selectSuggestion(index);
        }
    }
    
    onDocumentClick(event) {
        if (!this.container.contains(event.target)) {
            this.hideSuggestions();
        }
    }
    
    selectNext() {
        if (this.selectedIndex < this.suggestions.length - 1) {
            this.selectedIndex++;
            this.highlightSelected();
        }
    }
    
    selectPrevious() {
        if (this.selectedIndex > 0) {
            this.selectedIndex--;
            this.highlightSelected();
        }
    }
    
    highlightSelected() {
        const items = this.suggestionsElement.querySelectorAll(`.${this.config.suggestionClass}`);
        
        // Remove active class from all items
        items.forEach(item => item.classList.remove(this.config.activeClass));
        
        // Add active class to selected item
        if (this.selectedIndex >= 0 && this.selectedIndex < items.length) {
            const selectedItem = items[this.selectedIndex];
            selectedItem.classList.add(this.config.activeClass);
            
            // Ensure the selected item is visible in the scrollable list
            const container = this.suggestionsElement;
            const itemTop = selectedItem.offsetTop;
            const itemBottom = itemTop + selectedItem.offsetHeight;
            const containerTop = container.scrollTop;
            const containerBottom = containerTop + container.offsetHeight;
            
            if (itemTop < containerTop) {
                container.scrollTop = itemTop;
            } else if (itemBottom > containerBottom) {
                container.scrollTop = itemBottom - container.offsetHeight;
            }
        }
    }
    
    selectSuggestion(index) {
        if (index >= 0 && index < this.suggestions.length) {
            const suggestion = this.suggestions[index];
            this.inputElement.value = suggestion;
            this.hideSuggestions();
            
            // Call the onSelect callback if provided
            if (typeof this.config.onSelect === 'function') {
                this.config.onSelect(suggestion);
            }
            
            // Trigger change event on the input
            const event = new Event('change', { bubbles: true });
            this.inputElement.dispatchEvent(event);
        }
    }
    
    async fetchSuggestions(query) {
        try {
            const url = `${this.config.endpoint}?query=${encodeURIComponent(query)}&max=${this.config.maxResults}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.suggestions = data.suggestions || [];
            
            if (this.suggestions.length > 0) {
                this.showSuggestions(query);
            } else {
                this.hideSuggestions();
            }
        } catch (error) {
            console.error('Error fetching autocomplete suggestions:', error);
            this.hideSuggestions();
        }
    }
    
    showSuggestions(query) {
        // Clear existing suggestions
        this.suggestionsElement.innerHTML = '';
        
        // Create list items for each suggestion
        this.suggestions.forEach((suggestion, index) => {
            const listItem = document.createElement('li');
            listItem.className = this.config.suggestionClass;
            
            // Highlight matching text if enabled
            if (this.config.highlightMatches) {
                const regex = new RegExp(`(${query.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')})`, 'gi');
                listItem.innerHTML = suggestion.replace(regex, '<strong>$1</strong>');
            } else {
                listItem.textContent = suggestion;
            }
            
            // Style the list item
            Object.assign(listItem.style, {
                padding: '8px 10px',
                cursor: 'pointer',
                borderBottom: '1px solid #eee'
            });
            
            // Add hover effect
            listItem.addEventListener('mouseenter', () => {
                this.selectedIndex = index;
                this.highlightSelected();
            });
            
            this.suggestionsElement.appendChild(listItem);
        });
        
        // Reset selected index
        this.selectedIndex = -1;
        
        // Show suggestions
        this.suggestionsElement.style.display = 'block';
    }
    
    hideSuggestions() {
        this.suggestionsElement.style.display = 'none';
        this.selectedIndex = -1;
    }
}

// CSS for the autocomplete components that will be added to the head
function addAutocompleteStyles() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .autocomplete-container {
            position: relative;
            width: 100%;
        }
        
        .autocomplete-suggestions {
            position: absolute;
            top: 100%;
            left: 0;
            z-index: 1000;
            width: 100%;
            max-height: 300px;
            overflow-y: auto;
            background-color: #fff;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 4px 4px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.175);
            padding: 0;
            margin: 0;
            list-style: none;
        }
        
        .autocomplete-suggestion {
            padding: 8px 10px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
        }
        
        .autocomplete-suggestion:last-child {
            border-bottom: none;
        }
        
        .autocomplete-suggestion:hover,
        .autocomplete-active {
            background-color: #f8f9fa;
        }
        
        .autocomplete-suggestion strong {
            font-weight: bold;
            color: #007bff;
        }
    `;
    document.head.appendChild(styleElement);
}

// Add styles when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    addAutocompleteStyles();
});