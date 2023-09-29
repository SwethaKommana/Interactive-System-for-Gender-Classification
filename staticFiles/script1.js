URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("btn1");
recordButton.addEventListener("click", startRecording);

function startRecording() {
	console.log("recordButton clicked");
    var constraints = { audio: true, video:false }
 	/*Disable the record button until we get a success or fail from getUserMedia()*/
	recordButton.disabled = true;
	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		/*create an audio context after getUserMedia is called sampleRate might change after getUserMedia is called, like it does on macOS when 
		recording through AirPods sampleRate defaults to the one set in your OS for your playback device */
		audioContext = new AudioContext();

		//update the format 
		document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

		/*  assign to gumStream for later use  */
		gumStream = stream;
		
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		/* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
		rec = new Recorder(input,{numChannels:1})

		//start the recording process
		rec.record()

		console.log("Recording started");

    setTimeout(() => {
      // this will trigger one final 'ondataavailable' event and set recorder state to 'inactive'
      console.log("Recording stopped automatically after 30sec");
      rec.stop();
      //stop microphone access
	    gumStream.getAudioTracks()[0].stop();
      //create the wav blob and pass it on to createDownloadLink
	    rec.exportWAV(createDownloadLink);
  },20000);
	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
	});
}
function createDownloadLink(blob) {
	var url = URL.createObjectURL(blob);
	var au = document.createElement('audio');
	var link = document.createElement('a');
  	const forml = document.createElement('form');
  	forml.setAttribute("method","post");
  	forml.setAttribute("id","form__submit")
  	forml.setAttribute("action","/upload");
  	//forml.setAttribute("action","/upload/<string:ffilename>");
	//name of .wav file to use during upload and download (without extendion)
	const d = new Date();
	var filename = d.toString();
	//add controls to the <audio> element
	au.controls = true;
	au.src = url;
	//save to disk link
	link.href = url;
	var filename = "destination";
	//var ffilename = 'audio-'+ filename.slice(0,25)+".wav";
	//link.setAttribute("name",ffilename);
	//link.download = ffilename; //download forces the browser to donwload the file using the  filename
	link.download = filename+".wav";
	link.click();
  	forml.appendChild(link);
  	document.body.appendChild(au);
  	document.body.appendChild(forml);
    setTimeout(() => {
      let form = document.getElementById("form__submit");
      form.submit();
  },5000);
}