function setSound()
  val = tonumber(io.open( "/Users/vlyubin/code/lauzhack/lauzhack/volume_val", "r" ):read())
  hs.audiodevice.defaultOutputDevice():setVolume(val)
end

hs.timer.doEvery(0.1, setSound)
