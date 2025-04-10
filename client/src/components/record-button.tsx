import { useState } from 'react';

function RecordButton() {
    const [transcript, setTranscript] = useState("");

    function startRecording() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({audio: true}).then(
                (stream) => {
                    const mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start(5000);
                }
            );
        }
    }

    return (
        <>
        {
            /*
            1. Ask user to start recording audio from some source
            2. Establish web socket with the server
            3. Send json to web server with message <type />
            4. Terminate the web socket when user stops recording
            5. Update transcript state as transcript gets updated.
            */
        }
        </>
    );
}

export default RecordButton;