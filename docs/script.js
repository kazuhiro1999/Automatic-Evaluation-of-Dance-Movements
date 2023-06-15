let config;

// Load config data

fetch('config.json')
  .then(response => response.json())
  .then(data => {
    config = data;
    initialize();
    loadVideo();
    loadRatings();
    updateStatus();
  })
  .catch((error) => {
    console.error('Error:', error);
  });

let ratings = {};
let currentDancer = 1;
let currentChapter = 1;


function initialize() {

    document.getElementById('prev').addEventListener('click', function() {
        saveRatings();
        if (currentDancer > 1) {
            currentDancer--;
        } else if (currentChapter > 1){
            currentDancer = config.id.length;
            currentChapter--;
        } else {
            
        }
        loadVideo();
        loadRatings();
        updateStatus();
    });

    document.getElementById('next').addEventListener('click', function() {
        saveRatings();
        if (currentDancer < config.id.length) {
            currentDancer++;
        } else if (currentChapter < Object.keys(config.chapter).length){
            // move next chapter when finish evaluation of last dancer
            currentDancer = 1;
            currentChapter++;
        } else {
            
        }
        loadVideo();
        loadRatings();
        updateStatus();
    });

    document.getElementById('panelButton').addEventListener('click', function() {
        document.getElementById('panel').classList.toggle('open');
        document.getElementById('video').style.width = '70%'; 
        document.getElementById('header').style.width = '70%'; 
    });

    document.getElementById('closePanel').addEventListener('click', function() {
        document.getElementById('panel').classList.remove('open');
        document.getElementById('video').style.width = '100%';
        document.getElementById('header').style.width = '100%';
    });

    document.getElementById('download').addEventListener('click', function() {
        download_data();
    });

    const ratingsContainer = document.getElementById('ratings-container');
    ratingsContainer.innerHTML = ''; // reset rating items
      
    config.labels.forEach(label => {
        // create new rating items 
        const div = document.createElement('div');
        div.className = 'rating-item';
        
        const labelElement = document.createElement('label');
        labelElement.textContent = label;
        labelElement.htmlFor = `select_${label}`;
        
        const select = document.createElement('select');
        select.id = `select_${label}`;
        select.name = label;
      
        // add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '---';
        defaultOption.selected = true;
        defaultOption.disabled = true;
        select.appendChild(defaultOption);  

        // add options
        for(let key in config.options) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = config.options[key];
            select.appendChild(option);
        }
      
        div.appendChild(labelElement);
        div.appendChild(select);
        ratingsContainer.appendChild(div); 
    });

    // load ratings from localStorage
    let savedRatings = localStorage.getItem('Ratings');
    if (savedRatings) {
        ratings = JSON.parse(savedRatings);
    }
}

// save current ratings
function saveRatings() {
    ratings[currentDancer] = ratings[currentDancer] || {};
    ratings[currentDancer][currentChapter] = ratings[currentDancer][currentChapter] || {};

    for (let i = 0; i < config.labels.length; i++) {
        let label = config.labels[i];
        let rating = document.getElementById(`select_${label}`).value;
        ratings[currentDancer][currentChapter][label] = rating;
    }
    // save ratings to localStorage
    localStorage.setItem('Ratings', JSON.stringify(ratings));
}

// load ratings
function loadRatings() {
    for (let i = 0; i < config.labels.length; i++) {
        let label = config.labels[i];
        let select = document.getElementById(`select_${label}`);
        select.value = (ratings[currentDancer] && ratings[currentDancer][currentChapter] && ratings[currentDancer][currentChapter][label]) || '';
    }
}

function updateStatus() {
    document.getElementById('status').textContent = `Dancer ${config.id[currentDancer - 1]}-${currentChapter}`;
}

function loadVideo() {
    let video = document.getElementById('video');
    video.src = config.urls[config.id[currentDancer - 1]];

    video.addEventListener('loadeddata', function() {
        video.currentTime = config.chapter[currentChapter.toString()][0];
    });

    video.addEventListener('timeupdate', function() {
        if (video.currentTime >= config.chapter[currentChapter.toString()][1]) {
            video.currentTime = config.chapter[currentChapter.toString()][0];
        }
    });
}

// download rating data
function download_data() {
    var fileName = "output.json";
    var jsonData = JSON.stringify(ratings);
    const link = document.createElement('a'); //create HTML link
    link.setAttribute('href', 'data:text/plain,' + encodeURIComponent(jsonData));
    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
}