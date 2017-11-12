oldval = -1

function setSound()
  status, myFile = pcall(function()
    return io.open( "/Users/vlyubin/code/lauzhack/lauzhack/volume_val", "r" )
  end)
  if not status then
    return
  end
  if myFile == nil then
    return
  end
  status, val = pcall(function()
    return tonumber(myFile:read())
  end)
  if not status then
    return
  end
  if val == nil then
    return
  end
  if val ~= oldval then
    hs.audiodevice.defaultOutputDevice():setVolume(val)
    oldval = val
  end
  myFile:close()
end

hs.hotkey.bind({"cmd", "ctrl"}, "W", function()
  hs.alert.show("Hello World!")
end)

hs.hotkey.bind({"cmd", "ctrl"}, "E", function()
  hs.alert.show("Activated HammerSpoon!")
  hs.timer.doEvery(0.2, setSound)
end)
