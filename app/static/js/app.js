// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture');
const startCameraBtn = document.getElementById('startCamera');
const registerBtn = document.getElementById('registerBtn');
const recognizeBtn = document.getElementById('recognizeBtn');
const registerForm = document.getElementById('registerForm');
const registerStatus = document.getElementById('registerStatus');
const recognitionResult = document.getElementById('recognitionResult');
const logsTable = document.getElementById('logsTable');
const cameraSelect = document.getElementById('cameraSelect'); // Get the new selector element
const cameraSelectorContainer = document.querySelector('.camera-selector'); // Get the selector container
const startLiveBtn = document.getElementById('startLiveBtn');
const stopLiveBtn = document.getElementById('stopLiveBtn');


// Global variables
let stream = null;
let isCameraOn = false;
let currentTab = 'register';

// New variables for camera selection
let cameras = [];
let currentCameraId = null; 

let recognizeLoop = null;
let isRecognizing = false;

// --- Core Logic ---

// Event Listeners
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Load available cameras (must happen before trying to start stream)
    await getCameras(); 

    // 2. Initialize tabs
    const tabEl = document.querySelector('button[data-bs-toggle="tab"]');
    if (tabEl) {
        tabEl.addEventListener('shown.bs.tab', (event) => {
            currentTab = event.target.getAttribute('aria-controls');
            updateUIForTab();
        });
    }

    // 3. Attach Event Listeners
    startCameraBtn.addEventListener('click', toggleCamera);
    captureBtn.addEventListener('click', captureImage);
    registerForm.addEventListener('submit', handleRegistration);
    recognizeBtn.addEventListener('click', recognizeFace);
    
    // Add event listener for camera changes
    cameraSelect?.addEventListener('change', () => {
        // Automatically restart camera when a new one is selected
        if (isCameraOn) {
            // No need to call stopCamera/startCamera explicitly here,
            // the logic in toggleCamera and startCamera will handle stopping/restarting.
            toggleCamera();
        }
    });

    // Initial UI update and log load
    updateUIForTab();
    loadLogs();
});

// Toggle camera on/off
async function toggleCamera() {
    if (isCameraOn) {
        stopCamera();
    } else {
        await startCamera();
    }
    updateUIForTab();
}

// --- New Camera Selection and Loading Functions ---

// Add this function to populate camera selector
async function getCameras() {
    try {
        // Request media access first to get actual camera labels
        // We use a dummy request just to get permission and fill labels
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        tempStream.getTracks().forEach(track => track.stop());
        
        // Now enumerate devices
        cameras = await navigator.mediaDevices.enumerateDevices();
        cameras = cameras.filter(device => device.kind === 'videoinput');
        
        cameraSelect.innerHTML = '';
        
        cameras.forEach((camera, index) => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            // Use the label if available, otherwise use a generic name
            option.text = camera.label || `Camera ${index + 1}`; 
            cameraSelect.appendChild(option);
        });
        
        // Set the currently selected camera ID
        currentCameraId = cameraSelect.value;
        
        // Show the camera selector if we have multiple cameras
        if (cameras.length > 1) {
            cameraSelectorContainer.style.display = 'block';
        } else {
            cameraSelectorContainer.style.display = 'none';
        }
        
        return cameras;
    } catch (err) {
        console.error('Error getting cameras:', err);
        cameraSelect.innerHTML = '<option value="" disabled selected>Error loading cameras</option>';
        showAlert('error', `Could not load camera list: ${err.message}`);
        return [];
    }
}

// Update the startCamera function
async function startCamera() {
    try {
        // Get the selected camera ID from the dropdown
        const deviceId = cameraSelect.value || null;
        
        // If the stream is already running and the selected ID hasn't changed, do nothing
        if (isCameraOn && deviceId === currentCameraId) {
            return;
        }

        const constraints = {
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                // Use the exact deviceId if one is selected, otherwise rely on browser default
                deviceId: deviceId ? { exact: deviceId } : undefined
            },
            audio: false 
        };

        // Stop any existing stream before starting a new one
        if (stream) {
            stopCamera(false); // Stop tracks but don't reset UI fully
        }

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        video.srcObject = stream;
        isCameraOn = true;
        currentCameraId = deviceId; // Update the tracking ID

        startCameraBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
        captureBtn.disabled = false;
        
        // Enable/disable buttons based on tab
        if (currentTab === 'register') {
            registerBtn.disabled = false;
        } else {
            recognizeBtn.disabled = false;
        }
         
        startLiveBtn.disabled = false;
        stopLiveBtn.disabled = false;
 

        // Re-enumerate devices in case labels were previously unavailable
        await getCameras();
        // Ensure the correct option is selected after re-enumeration
        if (deviceId) {
            cameraSelect.value = deviceId;
        }

    } catch (err) {
        console.error('Error accessing camera:', err);
        showAlert('error', `Could not access camera: ${err.message}`);
        
        // If start fails, ensure everything is off
        stopCamera();
    }
}

// Update the stopCamera function
function stopCamera(resetUI = true) {
    
    startLiveBtn.disabled = true;
    stopLiveBtn.disabled = true;

    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        stream = null;
    }
    
    if (resetUI) {
        video.srcObject = null;
        isCameraOn = false;
        startCameraBtn.innerHTML = '<i class="fas fa-camera"></i> Start Camera';
        captureBtn.disabled = true;
        registerBtn.disabled = true;
        recognizeBtn.disabled = true;
    }
}

// --- Original Application Functions ---

// Capture image from video
function captureImage() {
    if (!isCameraOn) return;
    
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Show a brief flash effect
    video.style.transition = 'opacity 0.3s';
    video.style.opacity = '0.5';
    setTimeout(() => {
        video.style.opacity = '1';
    }, 200);
    //The image is converted to a base64-encoded JPEG string using canvas.toDataURL('image/jpeg', 0.9)
   //The JPEG quality is set to 0.9 (90% quality)
    return canvas.toDataURL('image/jpeg', 0.9);
}

// Handle face registration
async function handleRegistration(e) {
    e.preventDefault();
    
    const name = document.getElementById('name').value.trim();
    if (!name) {
        showAlert('error', 'Please enter a name', registerStatus);
        return;
    }
    
    const imageData = captureImage();
    if (!imageData) {
        showAlert('error', 'Could not capture image', registerStatus);
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('name', name);
    
    // Convert base64 to blob
    const blob = await (await fetch(imageData)).blob();
    formData.append('image', blob, 'face.jpg');
    
    // Disable button and show loading
    const originalText = registerBtn.innerHTML;
    registerBtn.disabled = true;
    registerBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Registering...';
    
    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showAlert('success', `Face registered successfully!`, registerStatus);
            document.getElementById('name').value = '';
            loadLogs(); // Refresh logs
        } else {
            throw new Error(result.detail || 'Failed to register face');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showAlert('error', `Error: ${error.message}`, registerStatus);
    } finally {
        registerBtn.disabled = false;
        registerBtn.innerHTML = originalText;
    }
}

// Recognize face
async function recognizeFace() {
    if (!isCameraOn) return;
    
    const imageData = captureImage();
    if (!imageData) {
        showAlert('error', 'Could not capture image', recognitionResult);
        return;
    }
    
    // Create form data
    const formData = new FormData();
    
    // Convert base64 to blob
    const blob = await (await fetch(imageData)).blob();
    formData.append('image', blob, 'face.jpg');
    
    // Disable button and show loading
    const originalText = recognizeBtn.innerHTML;
    recognizeBtn.disabled = true;
    recognizeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Recognizing...';
    
    try {
        const response = await fetch('/api/recognize', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            if (result.recognized && result.user) {
                const confidence = (result.user.confidence * 100).toFixed(2);
                showAlert('success', `Recognized: ${result.user.name} (${confidence}% confidence)`, recognitionResult);
            } else {
                showAlert('warning', 'No matching face found', recognitionResult);
            }
            loadLogs(); // Refresh logs
        } else {
            throw new Error(result.detail || 'Recognition failed');
        }
    } catch (error) {
        console.error('Recognition error:', error);
        showAlert('error', `Error: ${error.message}`, recognitionResult);
    } finally {
        recognizeBtn.disabled = false;
        recognizeBtn.innerHTML = originalText.replace('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Recognizing...', '<i class="fas fa-search"></i> Recognize Face');
    }
}

// Start continuous recognition
async function startContinuousRecognition(intervalMs = 1000) {
    if (isRecognizing) return;
    isRecognizing = true;

    if (!isCameraOn) await startCamera();

    const resultBox = document.getElementById('recognitionResult');
    showAlert('info', 'Recognition mode started...', resultBox);

    recognizeLoop = setInterval(async () => {
        try {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.8));

            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            const response = await fetch('/api/recognize', { method: 'POST', body: formData });
            const result = await response.json();

            if (response.ok && result.recognized && result.user) {
                const confidence = (result.user.confidence * 100).toFixed(2);
                showAlert('success', `✅ Recognized: ${result.user.name} (${confidence}% confidence)`, resultBox);
            } else {
                showAlert('warning', 'No match found', resultBox);
            }
        } catch (err) {
            console.error('Recognition loop error:', err);
            showAlert('error', 'Recognition error', resultBox);
        }
    }, intervalMs);
}

// Stop continuous recognition
function stopContinuousRecognition() {
    if (recognizeLoop) clearInterval(recognizeLoop);
    recognizeLoop = null;
    isRecognizing = false;
    showAlert('info', 'Recognition mode stopped');
}


// Load access logs
async function loadLogs() {
    try {
        const response = await fetch('/api/logs?limit=5');
        const result = await response.json();
        
        if (response.ok && result.logs && result.logs.length > 0) {
            renderLogs(result.logs);
        } else {
            logsTable.innerHTML = '<tr><td colspan="4" class="text-center">No logs available</td></tr>';
        }
    } catch (error) {
        console.error('Error loading logs:', error);
    }
}

// Render logs to table
function renderLogs(logs) {
    logsTable.innerHTML = '';
    
    logs.forEach(log => {
        const row = document.createElement('tr');
        
        // Format timestamp
        const date = new Date(log.timestamp);
        const timeString = date.toLocaleTimeString();
        
        // Status badge
        const statusClass = log.access_granted ? 'success' : 'danger';
        const statusText = log.access_granted ? 'Granted' : 'Denied';
        
        row.innerHTML = `
            <td>${timeString}</td>
            <td>${log.user ? log.user.name : 'Unknown'}</td>
            <td><span class="badge bg-${statusClass}">${statusText}</span></td>
            <td>${log.confidence ? (log.confidence * 100).toFixed(2) + '%' : 'N/A'}</td>
        `;
        
        logsTable.appendChild(row);
    });
}

// Update UI based on active tab
function updateUIForTab() {
    const isRegisterTab = currentTab === 'register';
    
    // Update button states
    captureBtn.style.display = isRegisterTab ? 'block' : 'none';
    
    // Reset forms and messages
    if (isRegisterTab) {
        document.getElementById('name').value = '';
        registerStatus.innerHTML = '';
        recognitionResult.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> Click 'Recognize Face' to identify a person
            </div>
        `;
    } else {
        recognitionResult.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> Click 'Recognize Face' to identify a person
            </div>
        `;
    }
}

// Show alert message
function showAlert(type, message, container) {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';
    
    const icon = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    }[type] || 'info-circle';
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        <i class="fas fa-${icon}"></i> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    if (container) {
        container.innerHTML = '';
        container.appendChild(alertDiv);
    } else {
        // If no container specified, show as toast
        const toastContainer = document.getElementById('toastContainer') || createToastContainer();
        toastContainer.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

// Create toast container if it doesn't exist
function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toastContainer';
    container.style.position = 'fixed';
    container.style.top = '20px';
    container.style.right = '20px';
    container.style.zIndex = '9999';
    container.style.width = '300px';
    document.body.appendChild(container);
    return container;
}

// Initialize Bootstrap tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize the application
function init() {
    initTooltips();
    updateUIForTab();
}

// Start the application when the DOM is fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Handle window resize
window.addEventListener('resize', () => {
    if (isCameraOn) {
        // Recalculate canvas dimensions
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
});

