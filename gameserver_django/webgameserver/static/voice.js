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