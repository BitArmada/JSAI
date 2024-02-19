

function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

function dsigmoid(z) {
    return sigmoid(z)*(1-sigmoid(z))
}

function softmax(arr) {
    const C = Math.max(...arr);
    const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
    return arr.map((value, index) => { 
        return Math.exp(value - C) / d;
    })
}

// mean squared error loss (p - t)**2
function MSE(p, t){
    var loss = 0;
    for(let i = 0; i < p.length; i++){
        for(let j = 0; j < p[0].length; j++){
            loss += (p[i][j]-t[i][j])**2
        }
    }
    loss/=(p.length*p[0].length)
    return loss;
};
// 2*(p - t)
function dMSE(p,t){
    var grad = matcomp(p,t, (p, t)=>{return 2*(p-t)})
    return grad;
};


// cross entropy
function xent(pred, target){
    var sum = 0;
    var soft = softmax(pred);
    for (var i = 0; i < soft.length; i++){
        sum += Math.log(soft[i])*target[i];
    }
    return -sum;
}

function dxent(pred, target){
    var grad = vectorize(softmax(pred));
    for(var i = 0; i < grad.length; i++){
        grad[i][0] -= target[i][0]
    }
    return grad
}

function matmult(m1, m2) {
    var result = [];
    for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m2[0].length; j++) {
            var sum = 0;
            for (var k = 0; k < m1[0].length; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

function vecmult(v1, v2){
    var out = [];
    for (var y = 0; y < v1.length; y++){
        var row = []
        for (var x = 0; x < v1[0].length; x++){
            row.push(v1[y][x]*v2[y][x])
        }
        out.push(row)
    }
    return out;
}

function matadd(v1, v2){
    var out = [];
    for (var y = 0; y < v1.length; y++){
        var row = []
        for (var x = 0; x < v1[0].length; x++){
            row.push(v1[y][x]+v2[y][x])
        }
        out.push(row)
    }
    return out;
}

function matcomp(v1, v2, func){
    var out = [];
    for (var y = 0; y < v1.length; y++){
        var row = []
        for (var x = 0; x < v1[0].length; x++){
            row.push(func(v1[y][x], v2[y][x]))
        }
        out.push(row)
    }
    return out;
}

function vectorize(array){
    var result = [];
    for(var i = 0; i < array.length; i ++){
        result.push([array[i]])
    }
    return result;
}

function matop(mat, func){
    var out = []
    for(var i = 0; i < mat.length; i++){
        var row = [];
        for(var j = 0; j < mat[0].length; j++){
            row.push(func(mat[i][j]))
        }
        out.push(row)
    }
    return out;
}

function matswap(mat){
    var out = []
    for(var x = 0; x < mat[0].length; x++){
        var row = [];
        for(var y = 0; y < mat.length; y++){
            row.push(mat[y][x])
        }
        out.push(row)
    }
    return out;
}

class Layer{
    constructor(size, activation){
        this.size = size;
        this.activation = activation;
        this.weights = [];
        this.bias = [];
        this.activationfunction;
        this.dactivationfunction;
        this.agradient = [];
        this.input = [];
        this.deltab = [];
        this.deltaw = [];
    }
    connect(connections){
        for(var i = 0; i < this.size; i++){
            var node = []
            for(var j = 0; j < connections; j++){
                node.push(Math.random()*2-1);
            }
            this.weights.push(node);
            this.bias.push([Math.random()*2-1]);
        }

        for(var j = 0; j < connections; j++){
            this.input.push([0])
        }

        this.deltab = matop(this.bias, (i)=>(0));
        this.deltaw = matop(this.weights, (i)=>(0));

        switch(this.activation){
            case "sigmoid":
                this.activationfunction = sigmoid;
                this.dactivationfunction = dsigmoid;
                break;
            case "xent":
                this.activationfunction = xent;
                this.dactivationfunction = dxent;
                break;
        }
    }
    predict(input){
        // copy input
        this.input = matop(input,(i)=>{return i});

        var result = matmult(this.weights, input);
        //add bias
        result = matadd(result, this.bias)
        //activate
        if(this.activationfunction){
            this.agradient = matop(result, this.dactivationfunction);
            result = matop(result, this.activationfunction);
        }

        return result;
    }
    propogate(gradient){
        // multiply da/do activation gradient with gradient vector
        if(this.activationfunction){
            gradient = vecmult(gradient, this.agradient);
        }

        // delta bias = -gradient * learning rate
        // this.bias = matadd(this.bias, matop(gradient, (i)=>{return i*(-k)}));
        this.deltab = matadd(this.deltab, gradient);

        // delta weights = -gradient * do/dw * learning rate
        const dodw = matmult(gradient, matswap(this.input))
        this.deltaw = matadd(this.deltaw, dodw);
        // this.weights = matadd(this.weights, matop(dodw, (i)=>{return i*(-k)}))

        // add do/di (how much input affects output) to the gradient vector
        // swap rows and cols of weights and multiply with gradient vector
        gradient = matmult(matswap(this.weights), gradient);
        
        // pass the gradient vector (delta loss) / (delta input to this layer) to the next layer
        return gradient;
    }

    applyDeltas(scaler){
        this.bias = matadd(this.bias, matop(this.deltab, (i)=>{return i*-scaler}));
        this.weights = matadd(this.weights, matop(this.deltaw, (i)=>{return i*-scaler}));

        // reset deltas
        this.deltab = matop(this.bias, (i)=>(0));
        this.deltaw = matop(this.weights, (i)=>(0));
    }
}

class Model{
    constructor(layers, lossfunction){
        this.layers = layers;
        this.lossfunction =  MSE;// MSE;
        this.dlossfunction = dMSE;
        this.compile();
    }
    compile(){
        for(var i = 1; i < this.layers.length; i++){
            this.layers[i].connect(this.layers[i-1].size)
        }
    }
    predict(input){
        var result = input;
        for(var i = 1; i < this.layers.length; i++){
            result = this.layers[i].predict(result);
        }
        return result;
    }
    backpropogate(input, target){
        var predicted = this.predict(input);
        var loss = this.lossfunction(predicted, target);
        var gradient = this.dlossfunction(predicted, target);
        console.log(gradient)
        for(var i = this.layers.length-1; i > 0; i--){
            gradient = this.layers[i].propogate(gradient);
        }
        return loss;
    }
    train(data){
        const LearningRate = 0.25;
        var loss = 0;
        for(var i = 0; i < data.length; i++){
            loss += this.backpropogate(data[i].in, data[i].out)
        }
        for(var i = this.layers.length-1; i > 0; i--){
            // apply average deltas
            this.layers[i].applyDeltas(LearningRate/data.length);
        }

        return loss / data.length;
    }
    batchTrain(data, batchsize){
        var loss = 0;
        var batches = 0;
        for (let i = 0; i < data.length; i += batchsize) {
            const batch = data.slice(i, i + batchsize);
            const l = this.train(batch);
            // console.log(Math.round((i/data.length)*100)+"% - loss: "+l)
            loss += l;
            batches++;
        }

        loss/=batches;

        return loss;
        
    }

    save(){
        return JSON.stringify(this);
    }
}

export {Model,Layer, vectorize, softmax}
