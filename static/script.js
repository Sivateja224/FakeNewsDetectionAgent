// script.js
// Client-side controller for the Fake News Detection Agent Web Dashboard.
// Handles API communication, UI updates, and current session history.

// Session history array to store predictions locally
let predictionHistory = [];

/**
 * Triggers the Machine Learning classification pipeline for the input text.
 */
function analyzeNews() {
    const textInput = document.getElementById("newsInput");
    const newsText = textInput.value.trim();
    
    // UI Elements
    const detectBtn = document.getElementById("detectBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    
    // 1. Input Validation
    if (!newsText) {
        alert("Please paste or type a news article first before analysis.");
        return;
    }
    
    // Check word count to guide the user (for optimal ML accuracy)
    const wordCount = newsText.split(/\s+/).filter(w => w.length > 0).length;
    if (wordCount < 5) {
        alert("Please enter a longer text (at least 5 words) for the classifier to analyze effectively.");
        return;
    }

    // 2. Set loading state in UI
    detectBtn.disabled = true;
    btnText.textContent = "Analyzing Article...";
    btnSpinner.classList.remove("d-none");
    
    // 3. Make AJAX POST request to the Flask server
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: newsText })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.message || "Server Error") });
        }
        return response.json();
    })
    .then(data => {
        if (data.status === "success") {
            // 4. Update Result Dashboard
            displayResults(data);
            
            // 5. Add to current session history
            addToHistory(newsText, data);
        } else {
            alert("Error: " + data.message);
        }
    })
    .catch(error => {
        console.error("Analysis failed:", error);
        alert(error.message || "An error occurred while communicating with the analysis agent. Make sure the model is trained.");
    })
    .finally(() => {
        // 6. Reset loading state
        detectBtn.disabled = false;
        btnText.innerHTML = '<i class="bi bi-search me-2"></i>Analyze & Detect';
        btnSpinner.classList.add("d-none");
    });
}

/**
 * Populates and animates the prediction result UI.
 */
function displayResults(data) {
    const promptState = document.getElementById("promptState");
    const resultState = document.getElementById("resultState");
    const resultBadge = document.getElementById("resultBadge");
    const confidenceVal = document.getElementById("confidenceVal");
    const confidenceBar = document.getElementById("confidenceBar");
    const explanationText = document.getElementById("explanationText");
    const patternCard = document.getElementById("patternCard");
    const patternList = document.getElementById("patternList");
    const recText = document.getElementById("recText");

    // Show result state and hide prompt placeholder
    promptState.classList.add("d-none");
    resultState.classList.remove("d-none");

    // Configure Badge UI and Progress Bar colors depending on Fake vs Real
    if (data.prediction_code === 1) { // Real News
        resultBadge.innerHTML = '<i class="bi bi-check-circle-fill me-2"></i>Real News';
        resultBadge.className = "d-inline-flex align-items-center px-4 py-2 rounded-pill fw-bold text-uppercase fs-5 shadow-sm bg-success text-white";
        
        confidenceBar.className = "progress-bar progress-bar-striped progress-bar-animated rounded-pill bg-success";
    } else { // Fake News
        resultBadge.innerHTML = '<i class="bi bi-exclamation-triangle-fill me-2"></i>Fake News';
        resultBadge.className = "d-inline-flex align-items-center px-4 py-2 rounded-pill fw-bold text-uppercase fs-5 shadow-sm bg-danger text-white";
        
        confidenceBar.className = "progress-bar progress-bar-striped progress-bar-animated rounded-pill bg-danger";
    }

    // Set texts and confidence scores
    confidenceVal.textContent = data.confidence + "%";
    
    // Animate progress bar transition
    confidenceBar.style.width = "0%";
    setTimeout(() => {
        confidenceBar.style.width = data.confidence + "%";
    }, 100);

    explanationText.textContent = data.explanation;
    recText.textContent = data.recommendation;

    // Handle suspicious pattern display
    patternList.innerHTML = "";
    if (data.suspicious_patterns && data.suspicious_patterns.length > 0) {
        patternCard.classList.remove("d-none");
        data.suspicious_patterns.forEach(pattern => {
            const li = document.createElement("li");
            li.className = "mb-1";
            li.innerHTML = `<i class="bi bi-arrow-right-short text-danger"></i> ${pattern}`;
            patternList.appendChild(li);
        });
    } else {
        patternCard.classList.add("d-none");
    }
}

/**
 * Resets the application's input and prediction UI views.
 */
function clearInput() {
    document.getElementById("newsInput").value = "";
    
    // Reset output panels to prompt state
    document.getElementById("promptState").classList.remove("d-none");
    document.getElementById("resultState").classList.add("d-none");
}

/**
 * Appends the latest prediction results to the session list.
 */
function addToHistory(originalText, responseData) {
    const emptyHistoryMsg = document.getElementById("emptyHistoryMsg");
    const historyList = document.getElementById("historyList");
    
    if (emptyHistoryMsg) {
        emptyHistoryMsg.classList.add("d-none");
    }

    // Create a truncated headline for history display
    let title = originalText.substring(0, 45);
    if (originalText.length > 45) title += "...";

    // Save prediction record
    const historyItem = {
        title: title,
        fullText: originalText,
        prediction: responseData.prediction,
        prediction_code: responseData.prediction_code,
        confidence: responseData.confidence,
        explanation: responseData.explanation,
        recommendation: responseData.recommendation,
        suspicious_patterns: responseData.suspicious_patterns,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };

    predictionHistory.unshift(historyItem); // Insert at beginning of list

    // Re-render history list
    renderHistory();
}

/**
 * Renders the session history list items.
 */
function renderHistory() {
    const historyList = document.getElementById("historyList");
    historyList.innerHTML = "";

    if (predictionHistory.length === 0) {
        historyList.innerHTML = `<p class="text-light-50 small text-center my-3" id="emptyHistoryMsg">No articles analyzed in this session yet.</p>`;
        return;
    }

    predictionHistory.forEach((item, index) => {
        const div = document.createElement("div");
        div.className = "history-item p-2 mb-2 rounded border text-white small d-flex justify-content-between align-items-center animate-fade-in";
        div.setAttribute("onclick", `selectHistoryItem(${index})`);
        
        const badgeColorClass = item.prediction_code === 1 ? "bg-success" : "bg-danger";
        
        div.innerHTML = `
            <div class="text-truncate me-2" style="max-width: 70%;">
                <strong>${escapeHtml(item.title)}</strong>
                <div class="text-light-50 text-xxs" style="font-size: 0.75rem;">${item.timestamp}</div>
            </div>
            <span class="badge ${badgeColorClass} text-uppercase px-2 py-1" style="font-size: 0.7rem;">
                ${item.prediction.split(' ')[0]} (${item.confidence}%)
            </span>
        `;
        historyList.appendChild(div);
    });
}

/**
 * Loads a historical prediction card immediately back onto the active workspace.
 */
function selectHistoryItem(index) {
    const item = predictionHistory[index];
    document.getElementById("newsInput").value = item.fullText;
    
    // Display results from cache instantly
    displayResults(item);
}

/**
 * Resets the session prediction list.
 */
function clearHistory() {
    predictionHistory = [];
    renderHistory();
}

/**
 * Helper to escape HTML tags to avoid script injections.
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}
