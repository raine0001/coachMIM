self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", (event) => {
  let payload = {};
  if (event.data) {
    try {
      payload = event.data.json();
    } catch (error) {
      payload = { title: "CoachMIM Alert", body: event.data.text() };
    }
  }

  const title = payload.title || "CoachMIM Alert";
  const options = {
    body: payload.body || "You have a new CoachMIM update.",
    icon: payload.icon || "/static/mim-logo.png",
    badge: payload.badge || "/static/mim-logo.png",
    tag: payload.tag || undefined,
    data: {
      url: payload.url || "/notifications",
      notificationId: payload.notificationId || null,
    },
    renotify: false,
    requireInteraction: false,
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || "/notifications";
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url === targetUrl && "focus" in client) {
          return client.focus();
        }
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow(targetUrl);
      }
      return null;
    })
  );
});
