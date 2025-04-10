import { useState } from 'react';
import { Button } from '@/components/ui/button'

function RecordButton() {
    const [transcript, setTranscript] = useState("");
    const [recording, setRecording] = useState(false);
    const socket = new WebSocket("http://localhost:8080");

    function startRecording() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({audio: true}).then(
                (stream) => {
                    const mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = (e) => {
                        if (socket.OPEN) {
                            const audio_chunk = new Blob([e.data], { type: "audio/ogg; codecs=opus"})
                            socket.send(audio_chunk)
                        }
                    };
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
            <Button>{recording ? "Stop Recording" : "Start Recording"}</Button>
        </>
    );
}

export default RecordButton;