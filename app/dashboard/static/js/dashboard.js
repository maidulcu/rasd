let cameras = [];

async function loadCameras() {
    const res = await fetch('/api/cameras');
    const data = await res.json();
    cameras = data.cameras;
    renderCameras();
}

function renderCameras() {
    const grid = document.getElementById('camera-grid');
    grid.innerHTML = cameras.length === 0 ? '<div class="no-cameras">No cameras added yet</div>' : '';

    cameras.forEach(cam => {
        const card = document.createElement('div');
        card.className = 'camera-card';
        card.innerHTML = `
            <div class="camera-preview">
                <div class="camera-placeholder">📹 ${cam.source.toUpperCase()}</div>
                <div class="camera-status">${cam.status}</div>
            </div>
            <div class="camera-info">
                <h4>${cam.name}</h4>
                <p class="camera-location">${cam.location || 'No location'}</p>
                <div class="camera-stats">
                    <span class="people">👥 ${cam.people_count}</span>
                </div>
                <div class="camera-actions">
                    <button class="btn btn-small" onclick="deleteCamera('${cam.id}')">Delete</button>
                </div>
            </div>
        `;
        grid.appendChild(card);
    });
}

function openAddCamera() {
    document.getElementById('add-camera-modal').classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

function showUpgrade() {
    document.getElementById('upgrade-modal').classList.add('active');
}

document.getElementById('add-camera-form').onsubmit = async (e) => {
    e.preventDefault();
    const data = {
        name: document.getElementById('camera-name').value,
        source: document.getElementById('camera-source').value,
        url: document.getElementById('camera-url').value,
        location: document.getElementById('camera-location').value,
    };
    await fetch('/api/cameras', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data),
    });
    closeModal('add-camera-modal');
    loadCameras();
    e.target.reset();
};

async function deleteCamera(id) {
    await fetch(`/api/cameras/${id}`, {method: 'DELETE'});
    loadCameras();
}

async function loadStats() {
    const res = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('cameras-online').textContent = `${data.online_cameras}/${data.total_cameras}`;
    document.getElementById('people-detected').textContent = data.total_people;
    document.getElementById('objects-count').textContent = data.total_objects;
    document.getElementById('fps-display').textContent = data.fps;
}

document.getElementById('connection-status').className = 'status online';
document.getElementById('connection-status').textContent = '● Basic Mode';

loadCameras();
setInterval(loadStats, 5000);
