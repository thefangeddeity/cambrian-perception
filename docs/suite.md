# The camera suite

cambrian-perception (the organism) and hls-livecam (the livecam) are two
products that share one camera, like two apps in one software suite. They
never run together:

- **Starting either stops the other.** The one that's stopped stays off,
  across reboots too, until someone starts it again.
- **At boot, whichever was on at shutdown comes back.**
- **Each side says why the other is off.** Its status says "off: the other
  has the camera", with the command that takes it back. While it runs, its
  page notes that the other is off.

Each side calls only the other's public command, never its internals. Either
product works alone; when only one is installed, the contract does nothing.

## The contract, per platform

| | Linux | Windows | macOS |
|---|---|---|---|
| Organism's switch | `cambrian.target` | task `cambrian-perception` | login agent `org.cambrian.perception` |
| Livecam's switch | `hls-livecam.target` | task `hls-livecam-win` | login agent `com.livecam.autostart` |
| Start (takes the camera) | `cambrian --start` / `camdash --start` | `cambrian --start` / `camdash --start` | `cambrian --start` / `livecam start` |
| Yield (stop, stay off) | `cambrian --yield` / `camdash --yield` | `cambrian --yield livecam` / `camdash --yield cambrian` | `cambrian --yield livecam` / `livecam yield cambrian` |
| Exclusion | systemd: the two targets `Conflicts=` each other | each start first calls the other's yield | the same |
| Off at boot | the yielded target is disabled | the yielded side has a `yielded` record in its state, and its boot start exits cleanly | the same |

On Linux a plain `systemctl start` of either target still stops the other (the
conflict). Only the start commands also change which one comes back at boot.

## What the organism watches

While it runs, the organism has the camera, so its home source is the camera
itself: `0` on Windows and macOS, a `/dev/v4l/by-id/...` path on Linux. It is
no longer the livecam's stream, because that stream is never up while the
organism runs.

On macOS only a process in the logged-in user's GUI session may use the
camera; a background daemon's open just hangs. So both products run as login
agents there. The first time the organism opens the camera, macOS asks on
screen to allow it.
