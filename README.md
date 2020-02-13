# Mirage

Player for VR180 videos.

VR180 videos are videos with a separate channel for each eye.

## Adjusting permissions

In [Delight VR](/static/cdn.delight-vr.com/1.6.8/dl8-0c65c31ebaf3b71dd6b6a452405e9011ff75d027.js) under `planDescs.video` remove all values from the `domains` array and set `unlimitedDomains` to `true`.

```json5
{
    "planDescs": {
        "video": {
            "adaptiveStreamingEnabled": true,
            "domains": [],
            "spatialAudioEnabled": true,
            "unlimitedDomains": true,
            "videoEnabled": true,
            "whiteLabelEnabled": true
        }
    }
}
```
