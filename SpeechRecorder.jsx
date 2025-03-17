import React, { useState, useRef } from 'react';

function SpeechRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  
  // We'll store the MediaRecorder and audio data in refs so they persist across re-renders
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    setTranscript("");
    setError("");

    // Ask the browser for permission to use the microphone
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);

      // Reset any previous chunks
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = event => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstart = () => {
        console.log("Recording started");
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      setError("Microphone access denied or not available.");
      console.error(err);
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.onstop = async () => {
      console.log("Recording stopped");
      setIsRecording(false);
      // Combine the recorded chunks into a single Blob
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
      // Send the Blob to the server for transcription
      await sendAudioToServer(audioBlob);
    };
  };

  const sendAudioToServer = async (audioBlob) => {
    // Use FormData to upload the Blob as a file
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.wav");

    try {
      const response = await fetch("http://localhost:5000/transcribe", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        setError(errData.error || "Transcription failed");
        return;
      }

      const data = await response.json();
      setTranscript(data.transcript || "");
    } catch (err) {
      setError("Error sending audio to server.");
      console.error(err);
    }
  };

  return (
    <div>
      <h1>Speech Recorder</h1>

      {isRecording ? (
        <button onClick={stopRecording}>Stop Recording</button>
      ) : (
        <button onClick={startRecording}>Start Recording</button>
      )}

      {transcript && (
        <div>
          <h2>Transcript:</h2>
          <p>{transcript}</p>
        </div>
      )}

      {error && (
        <div style={{ color: 'red' }}>
          <h2>Error:</h2>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}

export default SpeechRecorder;
