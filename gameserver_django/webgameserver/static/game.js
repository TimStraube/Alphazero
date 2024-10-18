document.addEventListener("DOMContentLoaded", function() {
    let moves_player = 0;
    let moves_ai = 0;
    let grid = document.getElementById("svg_grid");
    let size = 9;
    let map_fields = size * size;
    let rectWidth;
    let rectHeight;
    let startX;
    let startY;   
    let rectX;
    let rectY;
    let horizontalPadding = 2;
    let verticalPadding = 2;
    let strokeWidth = 2;
    let fill_opacity = 0.1;

    function sendSvgSourceToPython(source) {
        fetch('/', {
            method: 'POST',
            headers:
            {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({Source : source})
        })
        .then(response => response.json())
            .then(data => {
                const json_data = JSON.parse(
                    JSON.stringify(data)
                );
                const state = json_data.State
                const player = json_data.Player
                const failed = json_data.Failed
                const message = json_data.Message

                if (~failed) {
                    updateLeftBoard(state)
                    updateRightBoard(state)
                }

                if (message == "You won") {
                    window.location.href = "won/"
                } 

                if (message == "AI won") {
                    window.location.href = "loss/"
                } 
            }
        )
    }

    function updateLeftBoard(state) 
    {
        for (let row = 0; row < size; row++) {
            for (let column = 0; column < size; column++) {
                // field has been hit
                if (state[2][row][column] == 255) {
                    dyn_fill_opacity = 0.7;
                    color = "red";
                // ship on field which has no been hit
                } else if (state[3][row][column] == 255) {
                    dyn_fill_opacity = 0.7;
                    color = "grey";
                // water
                } else if (state[1][row][column] == 255) {
                    dyn_fill_opacity = 0.7;
                    color = "blue";
                } else {
                    dyn_fill_opacity = 0.0;
                    color = "blue";
                }
                let rect_now = document.getElementById(
                    new String(row) +
                    new String(column)
                )
                rect_now.setAttribute(
                    "style", 
                    "fill: " + color + "; stroke: black; stroke-width: " + strokeWidth + "; fill-opacity:  " + dyn_fill_opacity + "; stroke-opacity: 1.0; font-family: 'EB Garamond'; font-size: 35px; "
                ) 
            }
        }
    }

    function updateRightBoard(state) 
    {
        for (let row = 0; row < size; row++) {
            for (let column = 0; column < size; column++) {
                if (state[5][row][column] == 255) {
                    dyn_fill_opacity = 0.7;
                    color = "red";
                } else if (state[4][row][column] == 255) {
                    dyn_fill_opacity = 0.7;
                    color = "blue";
                } else {
                    dyn_fill_opacity = 0.0;
                    color = "blue";
                }
                let rect_now = document.getElementById(
                    new String(row) +
                    new String(column) + 
                    new String(map_fields)
                )
                
                rect_now.setAttribute(
                    "style", 
                    "fill: " + color + "; stroke: black; stroke-width: " + strokeWidth + "; fill-opacity:  " + dyn_fill_opacity + "; stroke-opacity: 1.0; font-family: 'EB Garamond'; font-size: 35px; "
                )
            }
        }
    }

    function zeichneBoards() 
    {
        rectWidth = 5
        rectHeight = 5
        
        startX = 5
        startY = 10

        rectX = startX

        for (let row = 0; row < size; row++) {
            let rectY = startY;
            for (let column = 0; column < size; column++) {
                
                let rect = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "rect"
                );
                let text = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "text"
                );

                rect.setAttribute(
                    'id', 
                    new String(row) +
                    new String(column)
                );
                rect.setAttribute(
                    "class",
                    "prevent-select"
                )

                rect.onclick = function ()
                {
                    var svgSource = text.textContent
                    sendSvgSourceToPython(svgSource)
                };

                text.setAttribute(
                    "x", 
                    rectX + 20 / size + "%"
                );
                text.setAttribute(
                    "y", 
                    rectY + 40 / size + "%"
                );
                text.setAttribute("font-size", "20");
                text.setAttribute("fill", "black");
                text.setAttribute("text-anchor", "middle");
                text.setAttribute(
                    "dominant-baseline", 
                    "middle"
                );
                text.textContent = (
                    String.fromCharCode(row + 65) +
                    (column + 1)
                );
                rect.setAttribute("x", rectX + "%");
                rect.setAttribute("y", rectY + "%");
                rect.setAttribute(
                    "width", 40 / size + "%"
                );
                rect.setAttribute(
                    "height", 80 / size + "%"
                );
                rect.setAttribute(
                    "style", 
                    "fill: grey; stroke: black; stroke-width: 2; fill-opacity:  " + fill_opacity + "; stroke-opacity: 1.0; font-family: 'EB Garamond'; font-size: 35px; "
                );
                // Rounded corners
                rect.setAttribute("rx", "1%");
                rect.setAttribute("ry", "1%");
                grid.appendChild(text);
                grid.appendChild(rect);
                rectY += 80 / size;
            }
            rectX += 40 / size;
        }
        
        rectWidth = 5
        rectHeight = 5
        
        rectX = 55

        startX = 10
        startY = 10
        
        for (let row = 0; row < size; row++) {
            let rectY = startY;
            for (let column = 0; column < size; column++) {
                let rect = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "rect");
                let text = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "text");
                rect.setAttribute(
                    'id', 
                    new String(row) + 
                    new String(column) + 
                    new String(map_fields));   
                rect.setAttribute(
                    "class",
                    "prevent-select"
                )
                rect.onclick = function () 
                {
                    var svgSource = text.textContent;
                    sendSvgSourceToPython(svgSource);
                };
                text.setAttribute(
                    "x", 
                    rectX + 20 / size + "%"
                );
                text.setAttribute(
                    "y", 
                    rectY + 40 / size + "%"
                );
                text.setAttribute("font-size", "20");
                text.setAttribute("fill", "black");
                text.setAttribute("text-anchor", "middle");
                text.setAttribute(
                    "dominant-baseline", 
                    "middle"
                );
                text.textContent = (
                    String.fromCharCode(row + 65) + 
                    (column + 1)
                );
                rect.setAttribute("x", rectX + "%");
                rect.setAttribute("y", rectY + "%");
                rect.setAttribute(
                    "width", 40 / size + "%"
                );
                rect.setAttribute(
                    "height", 80 / size + "%"
                );
                rect.setAttribute(
                    "style", 
                    "fill: blue; stroke: black; stroke-width: 2; fill-opacity: 0.1; stroke-opacity: 1.0; font-family: 'EB Garamond'; font-size: 35px; "
                );
                // Rounded corners
                rect.setAttribute("rx", "1%");
                rect.setAttribute("ry", "1%");
                grid.appendChild(text);
                grid.appendChild(rect);

                rectY += 80 / size;
            }
            rectX += 40 / size;
        }

        // Resize the grid to fit its containing rectangles
        if (window.innerWidth > window.innerHeight) 
        {
            gameFieldWidth = startX + size * (
                horizontalPadding + rectWidth + strokeWidth
            );
            gameFieldHeight = startY + size * (
                verticalPadding + rectHeight + strokeWidth
            );
        } else 
        {
            gameFieldWidth = startX + size * (
                horizontalPadding + rectWidth + strokeWidth
            );
            gameFieldHeight = startY + size * (
                verticalPadding + rectHeight + strokeWidth
            );
        }

        grid.setAttribute("width", gameFieldWidth);
        grid.setAttribute("height", gameFieldHeight);
    }

    zeichneBoards();

    var speechElement = new webkitSpeechRecognition();
    speechElement.lang = 'de-DE';
    speechElement.interimResults = true;
    speechElement.continuous = true;
    var final_transcript = '';
    var removeCharsInterval = null;
    speechElement.start();

    function removeFirstChar()
    {
        final_transcript = final_transcript.substring(1);
        document.getElementById('final').innerHTML = final_transcript;
        console.log(final_transcript);
        if (final_transcript.length < 1)
        {
            clearInterval(removeCharsInterval);
        }
        return;
    }

    speechElement.onresult = function(event) 
    {
        var interim_transcript = '';
        for(var i = event.resultIndex; i < event.results.length; ++i) 
        {
            if (event.results[i].isFinal) 
            {
                final_transcript += event.results[i][0].transcript;
                for (let ii = 65; ii <= (65 + size - 1); ii++) { 
                    console.log(ii)
                    let letter = String.fromCharCode(ii).toUpperCase();
                    for (let number = 1; number <= size; number++) {
                        let position = letter + number;
                        if (final_transcript.includes(
                            position
                        )) 
                        {
                            sendSvgSourceToPython(position);
                        }
                    }
                }
                if (final_transcript.includes("your")) 
                {
                    sendSvgSourceToPython("A1");
                }
                if (final_transcript.includes(
                    "white"
                )) 
                {
                    updateZustand(0)
                }
                if (final_transcript.includes(
                    "engage"
                )) 
                {
                    updateZustand(1)
                }
                removeCharsInterval = setInterval(
                    removeFirstChar, 
                    100
                );
            } else
            {
                interim_transcript += event.results[i][0].transcript;
                console.log(interim_transcript)
            }
        }
        document.getElementById('final').innerHTML = final_transcript;
        document.getElementById('interim').innerHTML = interim_transcript;
    }
});