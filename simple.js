

import {Model, Layer} from "./Model.js"

var canv = document.getElementById("canvas");
var ctx = canv.getContext('2d');
canv.width = 500;
canv.height = 500;

var classifier = new Model([
    new Layer(2),
    new Layer(8),
    new Layer(8, "sigmoid"),
    new Layer(2, "sigmoid"),
    new Layer(1, "sigmoid"),
], "MSE")

var data = [
    {in: [[1],[0]], out:[[1]]},
    {in: [[0],[1]], out:[[0]]},
    {in: [[0],[0]], out:[[0]]},
    {in: [[1],[1]], out:[[0]]}
]

classifier.predict([[0],[1]]);

function step(){
    var loss = classifier.train(data);
    console.log("loss: "+loss)
    for(var x = 0; x < 1; x+=0.1){
        for(var y = 0; y < 1; y+=0.1){
            const output = classifier.predict([[x],[y]]);
            // console.log(...output);
            ctx.fillStyle = `rgb(${output*255},${output*255},${output*255})`;
            ctx.fillRect(x*canv.width,y*canv.height,canv.width/10,canv.height/10);
        }
    }
}

setInterval(step, 1000/10);
// console.log(classifier.predict([[0],[1]]))
