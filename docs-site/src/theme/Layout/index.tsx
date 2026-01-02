import React from 'react';
import Layout from '@theme-original/Layout';
import ChatbotWidget from '@site/src/components/ChatbotWidget';

type Props = {
  children?: React.ReactNode;
};

export default function LayoutWrapper(props: Props): React.ReactElement {
  return (
    <>
      <Layout {...props}>
        {props.children}
        <ChatbotWidget />
      </Layout>
    </>
  );
}