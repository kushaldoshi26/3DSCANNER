# Social login setup

The Google and Facebook server flows are installed in `D:\3dSCANNER\api\oauth.py`. Both remain unavailable until configured. No credentials are included in source control.

Set these environment variables for the process that launches the scanner:

- `PUBLIC_BASE_URL`: the exact browser origin, for example `http://localhost:8000` for local Google testing. Production uses HTTPS.
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`: a Google web application OAuth client.
- `FACEBOOK_APP_ID` and `FACEBOOK_APP_SECRET`: a Meta app with Facebook Login enabled.
- `FACEBOOK_GRAPH_VERSION`: a supported Graph API version selected in your Meta app dashboard, including its `v` prefix.

Register these exact redirect URLs with the respective provider, replacing the origin for a deployed installation:

- `http://localhost:8000/api/auth/oauth/google/callback`
- `http://localhost:8000/api/auth/oauth/facebook/callback`

Restart the scanner after setting environment variables. Check `/api/auth/oauth/providers` for availability. Provider app review, test-user restrictions, domain requirements, and release approval still apply. Never paste app secrets into chat or commit them to Git.

Social identities use the provider's immutable account identifier. They create separate non-admin scanner accounts; accounts are not automatically merged by email. Sessions use the scanner's existing in-memory store, so a server restart signs users out.

References: [Google web server OAuth](https://developers.google.com/identity/protocols/oauth2/web-server), [Google userinfo](https://developers.google.com/identity/openid-connect/reference), [Meta manual flow](https://developers.facebook.com/docs/facebook-login/guides/advanced/manual-flow/).
