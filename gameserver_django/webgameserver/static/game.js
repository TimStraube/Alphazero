class Game {
    constructor(size) {
        // player 0 and 3 as indices for map
        this.rows = size;
        this.columns = size;
        this.size = size;
        this.actions = this.columns * this.rows;
        this.moves = 0;
    }

    toString() {
        return "battleship";
    }

    restart(player) {
        this.repeat = false;
        this.ships_possible = [[3, 2], [3, 2]];
        this.num_shipparts = this.ships_possible[0].reduce((a, b) => a + b, 0);
        // initialization of all submaps
        let state = Array.from({ length: 6 }, () => Array.from({ length: this.columns }, () => Array(this.rows).fill(0)));
        this.ships = [[], []];
        this.placeShips(state, player);
        this.placeShips(state, -player);
        return state;
    }

    shipIndex(player) {
        // f: {-1, 1} -> {0, 3}
        return 3 * (player > 0 ? 1 : 0);
    }

    hitIndex(player) {
        // f: {-1, 1} -> {1, 4}
        return 3 * (player > 0 ? 1 : 0) + 1;
    }

    knowledgeIndex(player) {
        // f: {-1, 1} -> {2, 5}
        return 3 * (player > 0 ? 1 : 0) + 2;
    }

    step(state, action, player) {
        let x = Math.floor(action / this.size);
        let y = action % this.size;

        let hit = state[this.hitIndex(player)][x][y];
        let ship = state[this.shipIndex(-player)][x][y];

        this.repeat = false;

        if (hit === 0 && ship === 0) {
            // hit water
            state[this.hitIndex(player)][x][y] = 255;
        } else if (hit === 0 && ship === 255) {
            // hit ship
            state[this.hitIndex(player)][x][y] = 255;
            state[this.knowledgeIndex(player)][x][y] = 255;
            this.repeat = true;
        } else {
            // already hit
        }
        return state;
    }

    pointsBetween(p1, p2) {
        let points = [];

        if (p1[0] === p2[0]) {
            let yValues = Array.from({ length: p2[1] - p1[1] + 1 }, (_, i) => i + p1[1]);
            points = yValues.map(y => [p1[0], y]);
        } else if (p1[1] === p2[1]) {
            let xValues = Array.from({ length: p2[0] - p1[0] + 1 }, (_, i) => i + p1[0]);
            points = xValues.map(x => [x, p1[1]]);
        }

        return points;
    }

    getValidMoves(state, player) {
        return state[this.hitIndex(player)].flat().map(val => val === 0 ? 1 : 0);
    }

    policy(policy, state) {
        let validMoves = state[this.hitIndex(1)].flat().map(val => val === 0 ? 1 : 0);
        policy = policy.map((val, idx) => val * validMoves[idx]);
        let sum = policy.reduce((a, b) => a + b, 0);
        policy = policy.map(val => val / sum);
        return policy;
    }

    checkWin(state, action, player) {
        let stateHit = state[this.hitIndex(player)];
        let stateShip = state[this.shipIndex(-player)];
        let hitSum = stateShip.flat().reduce((sum, val, idx) => sum + (val * stateHit.flat()[idx]), 0);
        return hitSum === this.num_shipparts;
    }

    terminated(state, action) {
        if (this.checkWin(state, action, 1)) {
            return [1, true];
        }
        if (this.checkWin(state, action, -1)) {
            return [1, true];
        }
        return [0, false];
    }

    changePerspective(state, player) {
        let returnState = Array.from({ length: 6 }, () => Array.from({ length: this.columns }, () => Array(this.rows).fill(0)));
        if (player === -1) {
            let stateCopy = state.slice(0, 3);
            returnState = state.slice(3, 6).concat(stateCopy);
            return returnState;
        } else {
            return state;
        }
    }

    getEncodedState(state) {
        let obsA = state.slice(this.hitIndex(1), this.knowledgeIndex(1) + 1).flat().map(val => val === 255 ? 1.0 : 0.0);
        let obsB = state.slice(this.hitIndex(-1), this.knowledgeIndex(-1) + 1).flat().map(val => val === 255 ? 1.0 : 0.0);
        let observation = new Float32Array([...obsB, ...obsA]);
        return observation;
    }

    placeShips(state, player) {
        for (let ship of this.ships_possible[player > 0 ? 1 : 0]) {
            let randomDirection = Math.floor(Math.random() * 2);

            let positions = [];

            for (let i = 0; i < this.size - ship + 1; i++) {
                let prefix = new Array(i).fill(0);
                let body = new Array(ship).fill(1);
                let postfix = new Array(this.size - ship - i).fill(0);
                let shipPossible = prefix.concat(body, postfix);

                let shipPossibleSqueezed;
                if (randomDirection) {
                    let shipMap = state[this.shipIndex(player)];
                    shipPossibleSqueezed = shipPossible.map((val, idx) => val && !shipMap[idx]);
                } else {
                    let transposedShipMap = this.transpose(state[this.shipIndex(player)]);
                    shipPossibleSqueezed = shipPossible.map((val, idx) => val && !transposedShipMap[idx]);
                }

                positions = positions.concat(shipPossibleSqueezed);
            }

            positions = this.reshape(positions, this.size - ship + 1, this.size);
            let possiblePositions = this.where(positions, 1);

            let lengthPossiblePositions = possiblePositions[0].length;

            let randomShipPosition = Math.floor(Math.random() * lengthPossiblePositions);

            let x = possiblePositions[0][randomShipPosition];
            let y = possiblePositions[1][randomShipPosition];

            let p1, p2;
            if (randomDirection) {
                p1 = [x, y];
                p2 = [x + ship - 1, y];
            } else {
                p1 = [y, x];
                p2 = [y, x + ship - 1];
            }

            let shipArray = this.pointsBetween(p1, p2);

            this.ships[player > 0 ? 1 : 0].push(shipArray);

            for (let point of shipArray) {
                state[this.shipIndex(player)][point[0]][point[1]] = 255;
            }
        }
    }

    transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    reshape(array, rows, cols) {
        let reshaped = [];
        for (let i = 0; i < rows; i++) {
            reshaped.push(array.slice(i * cols, (i + 1) * cols));
        }
        return reshaped;
    }

    where(array, value) {
        let indices = [[], []];
        for (let i = 0; i < array.length; i++) {
            for (let j = 0; j < array[i].length; j++) {
                if (array[i][j] === value) {
                    indices[0].push(i);
                    indices[1].push(j);
                }
            }
        }
        return indices;
    }
}

document.addEventListener("DOMContentLoaded", function() {
    let moves_player = 0
    let moves_ai = 0
    let grid = document.getElementById("svg_grid")
    let size = 9
    let map_fields = size * size
    let rectWidth
    let rectHeight
    let startX
    let startY   
    let rectX
    let rectY
    let horizontalPadding = 2
    let verticalPadding = 2
    let strokeWidth = 2
    let fill_opacity = 0.1
    let state = new Float32Array(1 * 6 * 9 * 9)

    const onnxSession = new onnx.InferenceSession()
    loadModel()

    function loadModel() {
        // load the ONNX model file
        onnxSession.loadModel("/static/models/model.onnx").then(() => {
            console.log("Model loaded successfully.");
        }).catch((error) => {
            console.error("Error during model loading:", error);
        });
    }

    function moveAi(state) {
        // generate model input
        const inputTensor = new onnx.Tensor(state, 'float32', [1, 4, 9, 9]);
        // execute the model
        onnxSession.run([inputTensor]).then((output) => {
            // log the output object
            console.log("Model output:", output);
            // consume the output
            const outputTensor = output.values().next().value;
            if (outputTensor) {
                console.log(`Model output tensor: ${outputTensor.data}.`);
                const predictedClass = outputTensor.data.indexOf(Math.max(...outputTensor.data));
                console.log(`Predicted class: ${predictedClass}`);
            } else {
                console.error("Model did not produce any output.");
            }
        }).catch((error) => {
            console.error("Error during model execution:", error);
        });
    }

    function getEncodedState(state) {
        // Define the hitIndex and knowledgeIndex functions
        function hitIndex(player) {
            // Implement the logic for hitIndex based on your requirements
            // This is a placeholder implementation
            return player === 1 ? 0 : 3 * 9 * 9;
        }
    
        function knowledgeIndex(player) {
            // Implement the logic for knowledgeIndex based on your requirements
            // This is a placeholder implementation
            return player === 1 ? 2 * 9 * 9 - 1 : 5 * 9 * 9 - 1;
        }
    
        // Extract the relevant slices from the input array
        const obsA = state.slice(hitIndex(1), knowledgeIndex(1) + 1).map(value => value === 255 ? 1.0 : 0.0);
        const obsB = state.slice(hitIndex(-1), knowledgeIndex(-1) + 1).map(value => value === 255 ? 1.0 : 0.0);
    
        // Concatenate the two resulting arrays
        const observation = new Float32Array([...obsB, ...obsA]);
    
        return observation;
    }
    

    function sendSvgSourceToPython(source) {
        fetch('/process_svg_source/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ Source: source })
        })
        .then(response => response.json())
        .then(data => {
            const state = data.State;
            const player = data.Player;
            const failed = data.Failed;
            const message = data.Message;
    
            if (!failed) {
                updateLeftBoard(state);
                updateRightBoard(state);
            }
    
            if (message === "You won") {
                window.location.href = "won/";
            } 
    
            if (message === "AI won") {
                window.location.href = "loss/";
            } 
        })
        .catch(error => console.error('Error:', error));
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
                    console.log(row, column)
                    encoded_state = getEncodedState(state)
                    moveAi(encoded_state)
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
                            // sendSvgSourceToPython(position);
                        }
                    }
                }
                if (final_transcript.includes("your")) 
                {
                    // sendSvgSourceToPython("A1");
                }
                if (final_transcript.includes(
                    "white"
                )) 
                {
                    // updateZustand(0)
                }
                if (final_transcript.includes(
                    "engage"
                )) 
                {
                    // updateZustand(1)
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