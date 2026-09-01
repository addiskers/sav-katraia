/**
 * MediaHandler: Manages Audio/Video capture and playback
 */
class MediaHandler {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.audioWorkletNode = null;
    this.videoStream = null;
    this.videoInterval = null;
    this.nextStartTime = 0;
    this.scheduledSources = [];
    this.isRecording = false;
    this.videoCanvas = document.createElement("canvas");
    this.canvasCtx = this.videoCanvas.getContext("2d");

    // Analyser nodes for visualization
    this.inputAnalyser = null;
    this.outputAnalyser = null;
    this.outputGain = null;

    // Half-duplex: gate mic uplink while agent PCM is playing (prevents echo STT).
    this.micGateUntil = 0;
    // Drop in-flight PCM briefly after a real barge-in interrupt.
    this.playbackMuted = false;
  }

  isMicGated() {
    if (!this.audioContext) return false;
    return this.audioContext.currentTime < (this.micGateUntil || 0);
  }

  async initializeAudio() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      await this.audioContext.audioWorklet.addModule(
        "/static/pcm-processor.js"
      );

      // Set up output analyser chain: analyser -> gain -> destination
      this.outputAnalyser = this.audioContext.createAnalyser();
      this.outputAnalyser.fftSize = 256;
      this.outputAnalyser.smoothingTimeConstant = 0.8;

      this.outputGain = this.audioContext.createGain();
      this.outputGain.gain.value = 1.0;

      this.outputAnalyser.connect(this.outputGain);
      this.outputGain.connect(this.audioContext.destination);
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
  }

  async startAudio(onAudioData) {
    await this.initializeAudio();

    try {
      // Echo cancellation is essential: without it the mic picks up the
      // agent's own TTS from the speakers and Deepgram transcribes it,
      // causing false barge-ins and the agent "hearing itself".
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const source = this.audioContext.createMediaStreamSource(
        this.mediaStream
      );

      // Set up input analyser
      this.inputAnalyser = this.audioContext.createAnalyser();
      this.inputAnalyser.fftSize = 256;
      this.inputAnalyser.smoothingTimeConstant = 0.8;
      source.connect(this.inputAnalyser);

      this.audioWorkletNode = new AudioWorkletNode(
        this.audioContext,
        "pcm-processor"
      );

      this.audioWorkletNode.port.onmessage = (event) => {
        if (this.isRecording) {
          const downsampled = this.downsampleBuffer(
            event.data,
            this.audioContext.sampleRate,
            16000
          );
          const pcm16 = this.convertFloat32ToInt16(downsampled);
          onAudioData(pcm16);
        }
      };

      // Route: source -> inputAnalyser -> worklet -> muteGain -> destination
      this.inputAnalyser.connect(this.audioWorkletNode);
      const muteGain = this.audioContext.createGain();
      muteGain.gain.value = 0;
      this.audioWorkletNode.connect(muteGain);
      muteGain.connect(this.audioContext.destination);

      this.isRecording = true;
    } catch (e) {
      console.error("Error starting audio:", e);
      throw e;
    }
  }

  stopAudio() {
    this.isRecording = false;
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.audioWorkletNode) {
      this.audioWorkletNode.disconnect();
      this.audioWorkletNode = null;
    }
    if (this.inputAnalyser) {
      this.inputAnalyser.disconnect();
      this.inputAnalyser = null;
    }
    if (this.outputAnalyser) {
      this.outputAnalyser.disconnect();
      this.outputAnalyser = null;
    }
    if (this.outputGain) {
      this.outputGain.disconnect();
      this.outputGain = null;
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.nextStartTime = 0;
    this.scheduledSources.forEach((s) => {
      try { s.stop(); s.disconnect(); } catch (e) {}
    });
    this.scheduledSources = [];
  }

  async startVideo(videoElement, onFrame) {
    try {
      this.videoStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      videoElement.srcObject = this.videoStream;

      this.videoInterval = setInterval(() => {
        this.captureFrame(videoElement, onFrame);
      }, 1000);
    } catch (e) {
      console.error("Error starting video:", e);
      throw e;
    }
  }

  async startScreen(videoElement, onFrame, onEnded) {
    try {
      this.videoStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
      });
      videoElement.srcObject = this.videoStream;

      this.videoStream.getVideoTracks()[0].onended = () => {
        this.stopVideo(videoElement);
        if (onEnded) onEnded();
      };

      this.videoInterval = setInterval(() => {
        this.captureFrame(videoElement, onFrame);
      }, 1000);
    } catch (e) {
      console.error("Error starting screen share:", e);
      throw e;
    }
  }

  stopVideo(videoElement) {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((t) => t.stop());
      this.videoStream = null;
    }
    if (this.videoInterval) {
      clearInterval(this.videoInterval);
      this.videoInterval = null;
    }
    if (videoElement) {
      videoElement.srcObject = null;
    }
  }

  captureFrame(videoElement, onFrame) {
    if (!this.videoStream) return;
    this.videoCanvas.width = 640;
    this.videoCanvas.height = 480;
    this.canvasCtx.drawImage(videoElement, 0, 0, 640, 480);
    const base64 = this.videoCanvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    onFrame(base64);
  }

  async playAudio(arrayBuffer) {
    if (!this.audioContext) return;
    // Drop audio chunks that arrive right after an interrupt
    if (this.playbackMuted) return;

    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    const pcmData = new Int16Array(arrayBuffer);
    const float32Data = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Data[i] = pcmData[i] / 32768.0;
    }

    const buffer = this.audioContext.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;

    // Route through output analyser for visualization
    if (this.outputAnalyser) {
      source.connect(this.outputAnalyser);
    } else {
      source.connect(this.audioContext.destination);
    }

    const now = this.audioContext.currentTime;
    this.nextStartTime = Math.max(now, this.nextStartTime);
    source.start(this.nextStartTime);
    const endAt = this.nextStartTime + buffer.duration;
    this.nextStartTime = endAt;
    // Keep mic muted until playback finishes (+ small tail for echo decay).
    this.micGateUntil = Math.max(this.micGateUntil || 0, endAt + 0.15);

    this.scheduledSources.push(source);
    source.onended = () => {
      const idx = this.scheduledSources.indexOf(source);
      if (idx > -1) this.scheduledSources.splice(idx, 1);
    };
  }

  stopAudioPlayback() {
    // Mute playback to drop any audio chunks still in-flight from server
    this.playbackMuted = true;
    this.micGateUntil = 0;
    if (this.outputGain) {
      try {
        this.outputGain.gain.cancelScheduledValues(this.audioContext.currentTime);
        this.outputGain.gain.setValueAtTime(0, this.audioContext.currentTime);
      } catch (e) {
        this.outputGain.gain.value = 0;
      }
    }

    this.scheduledSources.forEach((s) => {
      try {
        s.stop();
        s.disconnect();
      } catch (e) {}
    });
    this.scheduledSources = [];
    if (this.audioContext) {
      this.nextStartTime = this.audioContext.currentTime;
    }

    // Un-mute after a short delay so new (post-interrupt) audio can play
    clearTimeout(this._unmuteTimer);
    this._unmuteTimer = setTimeout(() => {
      this.playbackMuted = false;
      if (this.outputGain) {
        try {
          this.outputGain.gain.setValueAtTime(1, this.audioContext.currentTime);
        } catch (e) {
          this.outputGain.gain.value = 1;
        }
      }
    }, 600);
  }

  getInputAnalyser() {
    return this.inputAnalyser;
  }

  getOutputAnalyser() {
    return this.outputAnalyser;
  }

  // Utils
  downsampleBuffer(buffer, sampleRate, outSampleRate) {
    if (outSampleRate === sampleRate) return buffer;
    const ratio = sampleRate / outSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0,
        count = 0;
      for (
        let i = offsetBuffer;
        i < nextOffsetBuffer && i < buffer.length;
        i++
      ) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = accum / count;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  convertFloat32ToInt16(buffer) {
    let l = buffer.length;
    const buf = new Int16Array(l);
    while (l--) {
      buf[l] = Math.min(1, Math.max(-1, buffer[l])) * 0x7fff;
    }
    return buf.buffer;
  }
}
