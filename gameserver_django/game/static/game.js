class Game {
    constructor(size) {
        this.nj = window.nj
        this.grid = document.getElementById("svg_grid")
        this.rectWidth
        this.rectHeight
        this.startX
        this.startY 
        this.horizontalPadding = 2
        this.verticalPadding = 2
        this.strokeWidth = 2
        this.fill_opacity = 0.1
        // player 0 and 3 as indices for map
        this.size = size
        this.actions = this.size * this.size
        this.moves = 0
        this.user = 0
        this.alphazero = 1
        this.player = 0
        this.shipsPossible = [[5, 4, 3, 2], [5, 4, 3, 2]]
        // 0 ship placement, 1 battle
        this.phase = 0
        this.north_tminus1 = 0
        this.east_tminus1 = 0
        this.userPlacedBow = false
        this.userManuelShipPlacement = true
        this.state_ships = Array.from(
            { length: 2 }, 
            () => Array.from(
                { length: this.size }, 
                () => Array(this.size).fill(0)
            )
        )
        this.state_hits = Array.from(
            { length: 2 }, 
            () => Array.from(
                { length: this.size }, 
                () => Array(this.size).fill(0)
            )
        )
        this.state_experiance = Array.from(
            { length: 2 }, 
            () => Array.from(
                { length: this.size }, 
                () => Array(this.size).fill(0)
            )
        )
        this.onnxSession = new onnx.InferenceSession()
        this.loadModel()
    }

    loadModel() {
        // load the ONNX model file
        this.onnxSession.loadModel("/static/models/model.onnx").then(() => {
            console.log("Model loaded successfully.")
        }).catch((error) => {
            console.error(
                "Error during model loading:", 
                error
            )
        })
    }

    restart() {
        // player 0 or 1
        this.repeat = false
        for (let ship of this.shipsPossible[this.player]) {
            if (ship > this.size) {
                throw new Error(
                    `Ship length ${ship} is larger than the board size ${this.size}`
                )
            }
        }
        this.num_shipparts = this.shipsPossible[0].reduce(
            (a, b) => a + b, 0
        )
        if (this.num_shipparts >= this.size * this.size) {
            throw new Error(
                `Number of ship parts ${this.num_shipparts} is not smaller than the board size squared ${this.size * this.size}`
            )
        }
        this.state_ships = Array.from(
            { length: 2 }, 
            () => Array.from(
                { length: this.size }, 
                () => Array(this.size).fill(0)
            )
        )
        this.state_hits = Array.from(
            { length: 2 }, 
            () => Array.from(
                { length: this.size }, 
                () => Array(this.size).fill(0)
            )
        )
        this.state_experiance = Array.from(
            { length: 2 }, 
            () => Array.from(
                { length: this.size }, 
                () => Array(this.size).fill(0)
            )
        )
        this.ships = [[], []]
        // this.placeShips()
        // this.player = this.player ^ 1
        // this.placeShips()
    }

    togglePlayer() {
        this.player = this.player ^ 1
    }

    setAlphazeroPlayer() {
        this.player = this.alphazero
    }

    setUserPlayer() {
        this.player = this.user
    }

    step(action) {
        let north = action % this.size
        let east = Math.floor(action / this.size)
        if (this.phase === 0) {
            console.log(north, east)
            if (this.userPlacedBow) {
                // Check if ships can be placed inbetween north_tminus1 and north and east_tminus1 and east
                let points = this.pointsBetween(
                    [this.east_tminus1, this.north_tminus1], 
                    [east, north]
                )
                for (let point of points) {
                    if (this.state_ships[this.player][point[0]][point[1]] === 255) {
                        this.userPlacedBow = false
                        return
                    }
                }
                let lengthShip = points.length
                if (!this.shipsPossible[this.user].includes(lengthShip)) {
                    this.userPlacedBow = false;
                    return;
                }
                for (let point of points) {
                    this.state_ships[this.player][point[0]][point[1]] = 255
                }
                this.shipsPossible[this.user] = this.shipsPossible[this.user].filter(ship => ship !== lengthShip);
                if (this.shipsPossible[this.user].length === 0) {
                    this.setAlphazeroPlayer()
                    this.placeShips()
                    this.setUserPlayer()
                    this.phase = 1
                    console.log(
                        "Phase changed to battle phase."
                    )
                }
            }
            this.north_tminus1 = north 
            this.east_tminus1 = east
            this.userPlacedBow = !this.userPlacedBow
        } else {
            let hit = this.state_hits[this.player][east][north]
            let ship = this.state_ships[this.player][east][north]
            this.repeat = false
            if (hit === 0 && ship === 0) {
                // hit water
                console.log(
                    this.player + " hit water at field north " + north + " east " + east
                )
                this.state_hits[this.player][east][north] = 255
                this.togglePlayer()
            } else if (hit === 0 && ship === 255) {
                // hit ship
                console.log(
                    this.player + " hit ship at field north " + north + " east " + east
                )
                this.state_hits[this.player][east][north] = 255
                this.state_experiance[this.player][east][north] = 255
                this.repeat = true
                encoded_state = this.getEncodedState()
                stepAlphazero(encoded_state)
            } else {
                // already hit
                console.log(
                    "Field north " + north + " east " + east + " has already been hit."
                )
                if (this.player === this.alphazero) {
                    encoded_state = this.getEncodedState()
                    stepAlphazero(encoded_state)
                }
            }
            updateLeftBoard()
            updateRightBoard()
        }
    }

    pointsBetween(p1, p2) {
        let points = []
        if (p1[0] === p2[0]) {
            // Vertical line
            let [start, end] = p1[1] < p2[1] ? [p1[1], p2[1]] : [p2[1], p1[1]]
            for (let y = start; y <= end; y++) {
                points.push([p1[0], y])
            }
        } else if (p1[1] === p2[1]) {
            // Horizontal line
            let [start, end] = p1[0] < p2[0] ? [p1[0], p2[0]] : [p2[0], p1[0]]
            for (let x = start; x <= end; x++) {
                points.push([x, p1[1]])
            }
        }
        return points;
    }

    getValidMoves() {
        return this.state_hits[this.player].flat().map(
            val => val === 0 ? 1 : 0
        )
    }

    policy(policy) {
        let validMoves = this.state_hits[player].flat().map(val => val === 0 ? 1 : 0)
        policy = policy.map((val, idx) => val * validMoves[idx])
        let sum = policy.reduce((a, b) => a + b, 0)
        policy = policy.map(val => val / sum)
        return policy
    }

    checkWin() {
        let stateHit = this.state_hits[this.player]
        let stateShip = this.state_ships[this.player ^ 1]
        let hitSum = stateShip.flat().reduce(
            (sum, val, idx) => sum + (val * stateHit.flat()[idx]), 0
        )
        return hitSum === this.num_shipparts
    }

    terminated() {
        if (this.checkWin(this.user)) {
            return [1, true]
        }
        if (this.checkWin(this.alphazero)) {
            return [1, true]
        }
        return [0, false]
    }

    changePerspective() {
        let returnState = Array.from(
            { length: 6 }, 
            () => Array.from(
                { length: this.columns }, 
                () => Array(this.rows).fill(0)
            )
        )
        if (this.player === -1) {
            let stateCopy = this.state.slice(0, 3)
            returnState = state.slice(3, 6).concat(
                stateCopy
            )
            return returnState
        } else {
            return state
        }
    }

    getEncodedState() {
        // state_ships user, state_hits user, state_ships alphazero, state_hits alphazero, state_experiance user, state_experiance alphazero
        let encodedState = []
        encodedState = encodedState.concat(
            ...this.state_ships[this.user].flat()
        )
        encodedState = encodedState.concat(
            ...this.state_hits[this.user].flat()
        )
        encodedState = encodedState.concat(
            ...this.state_ships[this.alphazero].flat()
        )
        encodedState = encodedState.concat(
            ...this.state_hits[this.alphazero].flat()
        )
        return new Float32Array(
            encodedState.map(
                val => val === 255 ? 1.0 : 0.0
            )
        )
    }
    
    placeShips() {
        console.log(this.player === this.alphazero)
        for (let ship of this.shipsPossible[0]) {
            let placed = false;
            while (!placed) {
                // true for horizontal, false for vertical
                let direction = Math.random() < 0.5 
                this.startX = Math.floor(
                    Math.random() * (
                        this.size - (direction ? ship : 1)
                    )
                )
                this.startY = Math.floor(
                    Math.random() * (
                        this.size - (direction ? 1 : ship)
                    )
                )
                // Check if the ship can be placed
                let canPlace = true;
                for (let i = 0; i < ship; i++) {
                    let x = this.startX + (
                        direction ? i : 0
                    )
                    let y = this.startY + (
                        direction ? 0 : i
                    )
                    if (this.state_ships[this.user][x][y] !== 0) {
                        canPlace = false;
                        break;
                    }
                }
    
                // Place the ship if possible
                if (canPlace) {
                    for (let i = 0; i < ship; i++) {
                        let x = startX + (direction ? i : 0)
                        let y = startY + (direction ? 0 : i)
                        this.state_ships[this.user][x][y] = 255
                    }
                    placed = true;
                }
            }
        }
    }

    // placeShips() {
    //     for (let ship of this.shipsPossible[this.player]) {
    //         let randomDirection = Math.floor(Math.random() * 2)

    //         let positions = []

    //         for (let i = 0; i < this.size - ship + 1; i++) {
    //             let prefix = new Array(i).fill(0)
    //             let body = new Array(ship).fill(1)
    //             let postfix = new Array(this.size - ship - i).fill(0)
    //             let shipPossible = prefix.concat(body, postfix)

    //             let shipPossibleSqueezed
    //             if (randomDirection) {
    //                 shipPossibleSqueezed = this.state_ships[this.player] * shipPossible

    //                 let shipMap = this.state_ships[this.player]
    //                 shipPossibleSqueezed = shipPossible.map((val, idx) => val && !shipMap[idx])
    //             } else {
    //                 let transposedShipMap = this.transpose(this.state_ships[this.player])
    //                 shipPossibleSqueezed = shipPossible.map((val, idx) => val && !transposedShipMap[idx])
    //             }
    //             positions = positions.concat(shipPossibleSqueezed)
    //         }

    //         positions = this.reshape(positions, this.size - ship + 1, this.size)
    //         let possiblePositions = this.where(positions, 1)

    //         let lengthPossiblePositions = possiblePositions[0].length

    //         let randomShipPosition = Math.floor(Math.random() * lengthPossiblePositions)

    //         let x = possiblePositions[0][randomShipPosition]
    //         let y = possiblePositions[1][randomShipPosition]

    //         let p1, p2
    //         if (randomDirection) {
    //             p1 = [x, y]
    //             p2 = [x + ship - 1, y]
    //         } else {
    //             p1 = [y, x]
    //             p2 = [y, x + ship - 1]
    //         }

    //         let shipArray = this.pointsBetween(p1, p2)

    //         this.ships[this.player].push(shipArray)

    //         for (let point of shipArray) {
    //             state_ships[player][point[0]][point[1]] = 255
    //         }
    //     }
    // }

    transpose(matrix) {
        return matrix[0].map(
            (_, colIndex) => matrix.map(
                row => row[colIndex]
            )
        )
    }

    reshape(array, rows, cols) {
        let reshaped = []
        for (let i = 0; i < rows; i++) {
            reshaped.push(array.slice(i * cols, (i + 1) * cols))
        }
        return reshaped;
    }

    where(array, value) {
        let indices = [[], []]
        for (let i = 0; i < array.length; i++) {
            for (let j = 0; j < array[i].length; j++) {
                if (array[i][j] === value) {
                    indices[0].push(i)
                    indices[1].push(j)
                }
            }
        }
        return indices
    }
    
    stepAlphazero(state) {
        // generate model input
        const inputTensor = new onnx.Tensor(
            state, 
            'float32', 
            [1, 4, 9, 9]
        )
        // execute the model
        game.onnxSession.run([inputTensor]).then(
            (output) => {
                // log the output object
                // console.log("Model output:", output);
                // consume the output
                const outputTensor = output.values().next().value;
                if (outputTensor) {
                    // console.log(`Model output tensor: ${outputTensor.data}.`);
                    const action = outputTensor.data.indexOf(
                        Math.max(
                            ...outputTensor.data
                        )
                    )
                    // console.log(`Action: ${action}`);
                    game.step(action, game.alphazero)
                    updateLeftBoard() 
                } else {
                    console.error("Model did not produce any output.");
                }
            }
        ).catch((error) => {
            console.error(
                "Error during model execution:", 
                error
            )
        })
    }

    updateLeftBoard() {
        for (let row = 0; row < size; row++) {
            for (let column = 0; column < size; column++) {
                // field has been hit
                console.log(JSON.stringify(game.state_ships))
                if (game.state_experiance[game.user][row][column] == 255) {
                    dyn_fill_opacity = 0.7
                    color = "red"
                // ship on field which has no been hit
                } else if (game.state_ships[game.user][row][column] == 255) {
                    dyn_fill_opacity = 0.7
                    color = "grey"
                // water
                } else if (game.state_hits[game.alphazero][row][column] == 255) {
                    dyn_fill_opacity = 0.7
                    color = "blue"
                } else {
                    dyn_fill_opacity = 0.0
                    color = "blue"
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

    updateRightBoard(state) {
        for (let row = 0; row < size; row++) {
            for (let column = 0; column < size; column++) {
                if (game.state_experiance[game.alphazero][row][column] == 255) {
                    dyn_fill_opacity = 0.7
                    color = "red"
                } else if (game.state_hits[game.user][row][column] == 255) {
                    dyn_fill_opacity = 0.7
                    color = "blue"
                } else {
                    dyn_fill_opacity = 0.0
                    color = "blue"
                }
                let rect_now = document.getElementById(
                    new String(row) +
                    new String(column) + 
                    new String(this.size * this.size)
                )
                
                rect_now.setAttribute(
                    "style", 
                    "fill: " + color + "; stroke: black; stroke-width: " + strokeWidth + "; fill-opacity:  " + dyn_fill_opacity + "; stroke-opacity: 1.0; font-family: 'EB Garamond'; font-size: 35px; "
                )
            }
        }
    }

    plotBoards() {
        this.rectWidth = 5
        this.rectHeight = 5
        
        this.startX = 5
        this.startY = 10

        let rectX = this.startX

        for (let row = 0; row < this.size; row++) {
            let rectY = this.startY
            for (let column = 0; column < this.size; column++) {
                
                let rect = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "rect"
                )
                let text = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "text"
                )

                rect.setAttribute(
                    'id', 
                    new String(row) +
                    new String(column)
                )
                rect.setAttribute(
                    "class",
                    "prevent-select"
                )

                rect.onclick = function ()
                {
                    if (this.phase === 0) {
                        this.step(
                            row * this.size + column, 
                            this.user
                        )
                        this.updateLeftBoard()
                        this.updateRightBoard()
                    } else {
                        var svgSource = text.textContent
                        this.step(
                            row * this.size + column, 
                            this.user
                        )
                        // console.log(row, column)
                        encoded_state = this.getEncodedState()
                        this.stepAlphazero(encoded_state)
                        this.updateLeftBoard()
                        this.updateRightBoard()
                    }
                }

                text.setAttribute(
                    "x", 
                    rectX + 20 / this.size + "%"
                )
                console.log(rectY + 40 / this.size + "%")
                text.setAttribute(
                    "y", 
                    rectY + 40 / this.size + "%"
                )
                text.setAttribute("font-size", "20")
                text.setAttribute("fill", "black")
                text.setAttribute("text-anchor", "middle")
                text.setAttribute(
                    "dominant-baseline", 
                    "middle"
                )
                text.textContent = (
                    String.fromCharCode(row + 65) +
                    (column + 1)
                )
                rect.setAttribute("x", rectX + "%")
                rect.setAttribute("y", rectY + "%")
                rect.setAttribute(
                    "width", 40 / this.size + "%"
                )
                rect.setAttribute(
                    "height", 80 / this.size + "%"
                )
                rect.setAttribute(
                    "style", 
                    "fill: grey; stroke: black; stroke-width: 2; fill-opacity: 0.0; stroke-opacity: 1.0; font-family: 'EB Garamond'; font-size: 35px; "
                )
                // Rounded corners
                rect.setAttribute("rx", "1%")
                rect.setAttribute("ry", "1%")
                this.grid.appendChild(text)
                this.grid.appendChild(rect)
                rectY += 80 / this.size
            }
            rectX += 40 / this.size
        }
        
        this.rectWidth = 5
        this.rectHeight = 5
        
        rectX = 55

        this.startX = 10
        this.startY = 10
        
        for (let row = 0; row < this.size; row++) {
            let rectY = this.startY
            for (let column = 0; column < this.size; column++) {
                let rect = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "rect"
                )
                let text = document.createElementNS(
                    "http://www.w3.org/2000/svg", 
                    "text"
                )
                rect.setAttribute(
                    'id', 
                    new String(row) + 
                    new String(column) + 
                    new String(this.size * this.size)
                )
                rect.setAttribute(
                    "class",
                    "prevent-select"
                )
                rect.onclick = function () {
                    if (this.phase === 0) {
                        this.step(
                            row * this.size + column, 
                            this.user
                        )
                        this.updateLeftBoard()
                        this.updateRightBoard()
                    } else {
                        var svgSource = text.textContent
                        this.step(
                            row * this.size + column, 
                            this.user
                        )
                        this.updateLeftBoard()
                        this.updateRightBoard()
                        // console.log(row, column)
                        encoded_state = game.getEncodedState()
                        this.stepAlphazero(encoded_state)
                        this.updateLeftBoard()
                        this.updateRightBoard()
                    }
                }
                text.setAttribute(
                    "x", 
                    rectX + 20 / this.size + "%"
                )
                text.setAttribute(
                    "y", 
                    rectY + 40 / this.size + "%"
                )
                text.setAttribute("font-size", "20");
                text.setAttribute("fill", "black");
                text.setAttribute("text-anchor", "middle");
                text.setAttribute(
                    "dominant-baseline", 
                    "middle"
                )
                text.textContent = (
                    String.fromCharCode(row + 65) + 
                    (column + 1)
                )
                rect.setAttribute("x", rectX + "%");
                rect.setAttribute("y", rectY + "%");
                rect.setAttribute(
                    "width", 40 / this.size + "%"
                )
                rect.setAttribute(
                    "height", 80 / this.size + "%"
                )
                rect.setAttribute(
                    "style", 
                    "fill: grey; stroke: black; stroke-width: 2; fill-opacity: 0.0; stroke-opacity: 1.0; font-family: 'EB Garamond'; font-size: 35px; "
                )
                // Rounded corners
                rect.setAttribute("rx", "1%")
                rect.setAttribute("ry", "1%")
                this.grid.appendChild(text)
                this.grid.appendChild(rect)
                rectY += 80 / this.size
            }
            rectX += 40 / this.size
        }

        let gameFieldWidth = 0;
        let gameFieldHeight = 0;

        if (window.innerWidth > window.innerHeight) {
            gameFieldWidth = this.startX + this.size * (
                this.horizontalPadding + this.rectWidth + this.strokeWidth
            )
            gameFieldHeight = this.startY + this.size * (
                this.verticalPadding + this.rectHeight + this.strokeWidth
            )
        } else {
            gameFieldWidth = this.startX + this.size * (
                this.horizontalPadding + this.rectWidth + this.strokeWidth
            )
            gameFieldHeight = this.startY + this.size * (
                this.verticalPadding + this.rectHeight + this.strokeWidth
            )
        }

        this.grid.setAttribute("width", gameFieldWidth)
        this.grid.setAttribute("height", gameFieldHeight)
    }
}

document.addEventListener("DOMContentLoaded", function() {
    var game = new Game(9)
    // Restart game to a state in which no ships are placed
    game.restart()
    // Draw the empty game boards
    game.plotBoards()

    // updateLeftBoard()
    // updateRightBoard()
})